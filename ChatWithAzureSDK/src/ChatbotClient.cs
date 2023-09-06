using Azure.AI.OpenAI;
using Azure;
using Python.Runtime;
using Microsoft.DeepDev;
using System.Text.Json;
using System.Diagnostics;

namespace ChatWithAzureSDK
{
    public class ChatbotClient
    {
        private static Uri openAIEndpoint = new(Environment.GetEnvironmentVariable("OpenAIEndpoint"));
        private static AzureKeyCredential openAICredential = new(Environment.GetEnvironmentVariable("OpenAIKey"));
        private static Uri embeddingEndpoint = new(Environment.GetEnvironmentVariable("EmbeddingEndpoint"));
        private static Uri searchEndpoint = new(Environment.GetEnvironmentVariable("SEARCH_ENDPOINT"));
        private static AzureKeyCredential searchCredential = new(Environment.GetEnvironmentVariable("SEARCH_ADMIN_API_KEY"));

        private const string SearchIndexName = "index-800-chunksperdoc";

        public string SendMessage(string query, Queue<ChatMessage> conversation)
        {
            Console.WriteLine($"Here's the new query\n");
            Console.WriteLine($"User: {query} \n");

            OpenAIClient openAIClient = new(openAIEndpoint, openAICredential);
            var modelName = "gpt-35-turbo";

            // Add System prompt including context
            string prompt = "You are an AI assistant who helps users answer questions based on given documents.  If they don't provide enough context, do not answer.";
            var chatCompletionsOptions = new ChatCompletionsOptions()
            {
                Messages =
                {
                    new ChatMessage(ChatRole.System, prompt),
                }
            };

            conversation.Enqueue(new(ChatRole.User, query));

            // Add all the history including user query in chatCompletionsOptions
            foreach (ChatMessage chatMessage in conversation)
            {
                chatCompletionsOptions.Messages.Add(chatMessage);
            }

            chatCompletionsOptions.AzureExtensionsOptions = new()
            {
                Extensions =
                {
                    new AzureCognitiveSearchChatExtensionConfiguration(AzureChatExtensionType.AzureCognitiveSearch, searchEndpoint, searchCredential, SearchIndexName)
                    {
                        QueryType = AzureCognitiveSearchQueryType.Vector,
                        EmbeddingEndpoint = embeddingEndpoint,
                        EmbeddingKey = openAICredential,
                        FieldMappingOptions = new AzureCognitiveSearchIndexFieldMappingOptions(){
                            ContentFieldNames = { "Content" },
                            VectorFieldNames = { "ContentVector" }
                        }
                    },
                }
            };

            Console.WriteLine($"Waiting for an Open AI response....\n-");
            ChatCompletions answers = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);

            var intentJson = GetIntentJson(answers.Choices[0].Message.AzureExtensionsContext.Messages[0].Content);
            conversation.Enqueue(new ChatMessage(ChatRole.Tool, intentJson));
            conversation.Enqueue(answers.Choices[0].Message);

            Console.WriteLine($"Search Query : {intentJson} \n\n");
            Console.WriteLine($"Open AI Response : \n {answers.Choices[0].Message.Content}");
            Console.WriteLine($"\n-------------------------------------------------------\n");

            return answers.Choices[0].Message.Content;
        }

        private string GetIntentJson(string contentJson)
        {
            var userQuery = "";
            using (JsonDocument document = JsonDocument.Parse(contentJson))
            {
                JsonElement root = document.RootElement;
                foreach (JsonProperty property in root.EnumerateObject())
                {
                    if(property.Name == "intent")
                    {
                        userQuery = $"{{{property}}}";
                    }
                }
            }
            return userQuery;
        }

        public async Task<string> SendMessageUsingSearch(string query, Queue<ChatMessage> conversation)
        {
            Console.WriteLine($"Here's the new query\n");
            Console.WriteLine($"User: {query} \n");
            // Create a Stopwatch instance
            Stopwatch stopwatch = new Stopwatch();
            // Start the stopwatch
            stopwatch.Start();

            var openAIClient = new OpenAIClient(openAIEndpoint, openAICredential);
            var modelName = "gpt-4-32k";
            var tokenLimit = 32768; // Token limit for gpt-4-32k
            var maxResponseTokens = 4000;

            var searchFunction = new FunctionDefinition
            {
                Name = "QueryAzureSDK",
                Parameters = BinaryData.FromObjectAsJson(
                    new
                    {
                        Type = "object",
                        Properties = new
                        {
                            Query = new
                            {
                                Type = "string",
                                Description = "The query string to search information about Azure SDK",
                            }
                        },
                        Required = new[] { "query" },
                    },
                    new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
            };

            var chatCompletionsOptions = new ChatCompletionsOptions
            {
                Functions = { searchFunction },
                FunctionCall = FunctionDefinition.Auto,
                MaxTokens = maxResponseTokens
            };

            conversation.Enqueue(new ChatMessage(ChatRole.User, query));

            var convHistoryTokens = await NumOfTokensFromMessages(conversation);
            while (convHistoryTokens + maxResponseTokens >= tokenLimit)
            {
                conversation.Dequeue();
                convHistoryTokens = await NumOfTokensFromMessages(conversation);
            }

            foreach (var chatMessage in conversation)
            {
                chatCompletionsOptions.Messages.Add(chatMessage);
            }

            Console.WriteLine($"Waiting for an Open AI response to get search query based on the chat history....");
            var response = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);

            var responseChoice = response.Value.Choices[0];
            if (responseChoice.FinishReason == CompletionsFinishReason.FunctionCall)
            {
                conversation.Enqueue(responseChoice.Message);

                if (responseChoice.Message.FunctionCall.Name == "QueryAzureSDK")
                {
                    var queryJson = responseChoice.Message.FunctionCall.Arguments;
                    string userQuery = ParseUserQueryFromJson(queryJson);

                    Console.WriteLine($"Search Query recived by Open AI - {userQuery}\n\nCalling Search to get the context...\n");
                    IEnumerable<string> context = VectorSearch.Search(userQuery, 16); // Max 16,000 Tokens - Getting 16 chucks where each chunk is equal or less than 1000

                    // Add System prompt including context
                    string prompt = "You are an AI assistant who helps users answer questions based on the following documents.  If they don't provide enough context, do not answer.\n\n" + string.Join("\n\n", context);

                    var promptTokens = await GetTokenLength(prompt);
                    var queryFunctionTokens = await GetTokenLength(responseChoice.Message.FunctionCall.Arguments);

                    while (promptTokens + queryFunctionTokens + convHistoryTokens + maxResponseTokens >= tokenLimit)
                    {
                        conversation.Dequeue();
                        convHistoryTokens = await NumOfTokensFromMessages(conversation);
                    }

                    chatCompletionsOptions = new ChatCompletionsOptions()
                    {
                        Messages =
                        {
                            new ChatMessage(ChatRole.System, prompt),
                        },
                        MaxTokens = maxResponseTokens
                    };

                    // Add all the history including user query in chatCompletionsOptions
                    foreach (ChatMessage chatMessage in conversation)
                    {
                        chatCompletionsOptions.Messages.Add(chatMessage);
                    }

                    TimeSpan elapsedTime = stopwatch.Elapsed;
                    Console.WriteLine($"Total time taken till now: {elapsedTime.TotalSeconds} seconds");

                    stopwatch = new Stopwatch();
                    stopwatch.Start();
                    Console.WriteLine($"Waiting for an Open AI response based on these documents....\n-");
                    response = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);
                    responseChoice = response.Value.Choices[0];

                    Console.WriteLine($"Open AI Response : \n {responseChoice.Message.Content}");

                    elapsedTime = stopwatch.Elapsed;
                    Console.WriteLine($"\nTotal time taken by open AI request: {elapsedTime.TotalSeconds} seconds \n ---------------------------------------------------------------------------------------\n");
                    conversation.Enqueue(responseChoice.Message);
                }
            }

            return responseChoice.Message.Content;
        }

        public async Task<string> SendMessageGPT4(string query, Queue<ChatMessage> conversation)
        {
            //LoadDocuments();
            Console.WriteLine($"Here's the new query\n");
            Console.WriteLine($"User: {query} \n");
            // Create a Stopwatch instance
            Stopwatch stopwatch = new Stopwatch();
            // Start the stopwatch
            stopwatch.Start();

            var openAIClient = new OpenAIClient(openAIEndpoint, openAICredential);
            var modelName = "gpt-4";
            var tokenLimit = 8192; // Token limit for gpt-4
            var maxResponseTokens = 1000;

            var searchFunction = new FunctionDefinition
            {
                Name = "QueryAzureSDK",
                Parameters = BinaryData.FromObjectAsJson(
                    new
                    {
                        Type = "object",
                        Properties = new
                        {
                            Query = new
                            {
                                Type = "string",
                                Description = "The query string to search information about Azure SDK",
                            }
                        },
                        Required = new[] { "query" },
                    },
                    new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
            };

            var chatCompletionsOptions = new ChatCompletionsOptions
            {
                Functions = { searchFunction },
                FunctionCall = FunctionDefinition.Auto,
                MaxTokens = maxResponseTokens
            };

            conversation.Enqueue(new ChatMessage(ChatRole.User, query));

            var convHistoryTokens = await NumOfTokensFromMessages(conversation);
            while (convHistoryTokens + maxResponseTokens >= tokenLimit)
            {
                conversation.Dequeue();
                convHistoryTokens = await NumOfTokensFromMessages(conversation);
            }

            foreach (var chatMessage in conversation)
            {
                chatCompletionsOptions.Messages.Add(chatMessage);
            }

            Console.WriteLine($"Waiting for an Open AI response to get search query based on the chat history....");
            var response = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);

            var responseChoice = response.Value.Choices[0];
            if (responseChoice.FinishReason == CompletionsFinishReason.FunctionCall)
            {
                conversation.Enqueue(responseChoice.Message);

                if (responseChoice.Message.FunctionCall.Name == "QueryAzureSDK")
                {
                    var queryJson = responseChoice.Message.FunctionCall.Arguments;
                    string userQuery = ParseUserQueryFromJson(queryJson);

                    Console.WriteLine($"Search Query recived by Open AI - {userQuery}\n\nCalling Search to get the context...\n");
                    IEnumerable<string> context = VectorSearch.Search(userQuery, 5); // Max 4000 Tokens - Getting 5 chucks where each chunk is equal or less than 800

                    // Add System prompt including context
                    string prompt = "You are an AI assistant who helps users answer questions based on the following documents.  If they don't provide enough context, do not answer.\n\n" + string.Join("\n\n", context);

                    var promptTokens = await GetTokenLength(prompt);
                    var queryFunctionTokens = await GetTokenLength(responseChoice.Message.FunctionCall.Arguments);

                    while (promptTokens + queryFunctionTokens + convHistoryTokens + maxResponseTokens >= tokenLimit)
                    {
                        conversation.Dequeue();
                        convHistoryTokens = await NumOfTokensFromMessages(conversation);
                    }

                    chatCompletionsOptions = new ChatCompletionsOptions()
                    {
                        Messages =
                        {
                            new ChatMessage(ChatRole.System, prompt),
                        },
                        MaxTokens = maxResponseTokens
                    };

                    // Add all the history including user query in chatCompletionsOptions
                    foreach (ChatMessage chatMessage in conversation)
                    {
                        chatCompletionsOptions.Messages.Add(chatMessage);
                    }

                    TimeSpan elapsedTime = stopwatch.Elapsed;
                    Console.WriteLine($"Total time taken till now: {elapsedTime.TotalSeconds} seconds");

                    stopwatch = new Stopwatch();
                    stopwatch.Start();
                    Console.WriteLine($"Waiting for an Open AI response based on these documents....\n-");
                    response = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);
                    responseChoice = response.Value.Choices[0];

                    Console.WriteLine($"Open AI Response : \n {responseChoice.Message.Content}");

                    elapsedTime = stopwatch.Elapsed;
                    Console.WriteLine($"\nTotal time taken by open AI request: {elapsedTime.TotalSeconds} seconds \n ---------------------------------------------------------------------------------------\n");
                    conversation.Enqueue(responseChoice.Message);
                }
            }

            return responseChoice.Message.Content;
        }

        private string ParseUserQueryFromJson(string queryJson)
        {
            var userQuery = "";
            using (JsonDocument document = JsonDocument.Parse(queryJson))
            {
                JsonElement root = document.RootElement;
                if (root.TryGetProperty("query", out JsonElement queryElement) && queryElement.ValueKind == JsonValueKind.String)
                {
                    userQuery = queryElement.GetString();
                }
            }
            return userQuery;
        }

        public async Task<string> SendMessageSaveChatHistory(string query, Queue<ChatMessage> conversation)
        {
            // LoadDocuments();

            OpenAIClient openAIClient = new(openAIEndpoint, openAICredential);
            var modelName = "gpt-4";
            var tokenLimit = 8192; // Token limit for gpt-4
            var maxResponseTokens = 1000;

            IEnumerable<string> context = VectorSearch.Search(query, 5); // Max 4000 Tokens - Getting 5 chucks where each chunk is equal or less than 800

            // Add System prompt including context
            string prompt = "You are an AI assistant who helps users answer questions based on the following documents.  If they don't provide enough context, do not answer.\n\n" + string.Join("\n\n", context);
            var chatCompletionsOptions = new ChatCompletionsOptions()
            {
                Messages =
                {
                    new ChatMessage(ChatRole.System, prompt),
                }
            };

            // Manage conversation
            conversation.Enqueue(new(ChatRole.User, query));

            var promptTokens = await GetTokenLength(prompt);
            var convHistoryTokens = await NumOfTokensFromMessages(conversation);

            while (promptTokens + convHistoryTokens + maxResponseTokens >= tokenLimit)
            {
                conversation.Dequeue();
                convHistoryTokens = await NumOfTokensFromMessages(conversation);
            }

            // Add all the history including user query in chatCompletionsOptions
            foreach (ChatMessage chatMessage in conversation)
            {
                chatCompletionsOptions.Messages.Add(chatMessage);
            }

            chatCompletionsOptions.MaxTokens = maxResponseTokens;

            Console.WriteLine($"Waiting for an Open AI response....\n-");
            ChatCompletions answers = openAIClient.GetChatCompletions(modelName, chatCompletionsOptions);

            conversation.Enqueue(answers.Choices[0].Message);

            Console.WriteLine($"Open AI Response : \n {answers.Choices[0].Message.Content}");

            return answers.Choices[0].Message.Content;
        }

        public string SendMessageQnA(string query)
        {
            OpenAIClient openAIClient = new(openAIEndpoint, openAICredential);
            var modelName = "gpt-4";

            IEnumerable<string> context = VectorSearch.Search(query, 5);

            string prompt = "You are an AI assistant who helps users answer questions based on the following documents.  If they don't provide enough context, do not answer.\n\n" + string.Join("\n\n", context);

            Console.WriteLine($"Waiting for an Open AI response....\n-");
            ChatCompletions answers =
                openAIClient.GetChatCompletions(
                    modelName,
                    new ChatCompletionsOptions()
                    {
                        Messages =
                {
                    new ChatMessage(ChatRole.System, prompt),
                    new ChatMessage(ChatRole.User, query)
                }
                    });

            Console.WriteLine($"Open AI Response : \n {answers.Choices[0].Message.Content}");

            return answers.Choices[0].Message.Content;
        }

        public static void LoadDocuments()
        {
            // Save the chunked documents in a directory
            // StartPythonEngine();

            // Pass the .jsonl file path to index chunked documents
            VectorSearch.IndexDocuments(@"C:\Users\shreja\Demo\ChatWithAzureSDK\ChatWithAzureSDK\src\JsonDocument\documents.jsonl");
        }

        public static void StartPythonEngine()
        {
            PythonEngine.Initialize(); // Initialize Python.NET

            using (Py.GIL()) // Acquire the Python Global Interpreter Lock
            {
                string folderPath = "C:\\Users\\shreja\\Demo\\ChatWithAzureSDK\\ChatWithAzureSDK\\src"; // Replace with the actual path

                // Execute Python code to modify sys.path
                PythonEngine.Exec(@$"import sys
sys.path.append(r'{folderPath}')");

                try
                {
                    dynamic python = Py.Import("prepare_data"); // Import the Python module

                    dynamic loadDocuments = python.load_documents; // Get the Python function
                    string result = loadDocuments("C:\\Users\\shreja\\Demo\\ChatWithAzureSDK\\ChatWithAzureSDK\\src\\AzureSDKDocs"); // Call the Python function

                    Console.WriteLine(result);

                    //dynamic loadDocuments = python.load_documents_test1; // Get the Python function
                    //string result = loadDocuments("C:\\Users\\shreja\\Demo\\ChatWithAzureSDK\\ChatWithAzureSDK\\src\\AzureSDKDocs"); // Call the Python function

                    //Console.WriteLine(result);

                    //Console.WriteLine("Testing..................!!!!!!");

                    //loadDocuments = python.load_documents_test2; // Get the Python function
                    //result = loadDocuments("C:\\Users\\shreja\\Demo\\ChatWithAzureSDK\\ChatWithAzureSDK\\src\\AzureSDKDocs"); // Call the Python function

                    //Console.WriteLine(result);
                }
                catch (PythonException ex)
                {
                    Console.WriteLine("Python Exception: " + ex.Message);
                }
            }

            PythonEngine.Shutdown(); // Clean up resources
        }

        public async Task<int> NumOfTokensFromMessages(Queue<ChatMessage> messages)
        {
            int numTokens = 0;
            foreach (var message in messages)
            {
                if (message.Content != null)
                {
                    var tokens = await GetTokenLength(message.Content);
                    numTokens += tokens;
                }
            }
            return numTokens;
        }

        public async Task<int> GetTokenLength(string text)
        {
            var tokenizer = await TokenizerBuilder.CreateByModelNameAsync("gpt-4-32k");
            IList<int> tokens = tokenizer.Encode(text, allowedSpecial: null);
            return tokens.Count;
        }
    }
}
