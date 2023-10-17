using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.AI.OpenAI;
using Azure.Search.Documents.Models;
using Azure.Search.Documents;
using Azure;
using System.Text.Json;

namespace ChatWithAzureSDK
{
    public partial class VectorSearch
    {
        private const string ModelName = "text-embedding-ada-002";
        private const int ModelDimensions = 1536;
        private const string VectorSearchIndexName = "index-800-chunksperdoc";
       // private const string SemanticVectorSearchIndexName = "semantic-index-800-chunksperdoc";

        private static Uri searchEndpoint = new(Environment.GetEnvironmentVariable("SEARCH_ENDPOINT"));
        private static AzureKeyCredential searchCredential = new(Environment.GetEnvironmentVariable("SEARCH_API_KEY"));

        private static Uri openAIEndpoint = new(Environment.GetEnvironmentVariable("OPENAI_ENDPOINT"));
        private static AzureKeyCredential openAICredential = new(Environment.GetEnvironmentVariable("OPENAI_KEY"));

        public static void IndexDocuments(string path = @"C:\Users\shreja\Demo\ChatWithAzureSDK\ChatWithAzureSDK\src\JsonDocument\documents.jsonl")
        {
            //-----------Create Index---------------------
            SearchIndexClient indexClient = new(searchEndpoint, searchCredential);

            SearchIndex index = GetIndex(VectorSearchIndexName);
            indexClient.CreateIndex(index);

            //--------Upload chunked documents------------
            UploadChunks(path);
        }

        internal static IEnumerable<string> Search(string query, int count)
        {
            SearchClient searchClient = new(searchEndpoint, VectorSearchIndexName, searchCredential);
            OpenAIClient openAIClient = new(openAIEndpoint, openAICredential);

            var vectorizedResult = Vectorize(openAIClient, query);

            var vector = new SearchQueryVector { Value = vectorizedResult, KNearestNeighborsCount = count, Fields = { "ContentVector" } };
            SearchResults<AzureSDKDocument> response = searchClient.Search<AzureSDKDocument>(
                   null,
                   new SearchOptions
                   {
                       Vectors = { vector },
                       Select = { "Id", "Content", "Source" }
                   });

            int resultCount = 0;
            foreach (SearchResult<AzureSDKDocument> result in response.GetResults())
            {
                string contentToPrint = result.Document.Content.Length > 20 ? result.Document.Content.Substring(0, 20) : result.Document.Content;
                Console.WriteLine($"Document {++resultCount} - \n Source - {result.Document.Source} \n Content - {contentToPrint}.\n.\n.\n.");
                Console.WriteLine($"------------------------------------------------------------- \n\n");
                yield return result.Document.Content;
            }
        }


        /// <summary> Get a <see cref="SearchIndex"/>. </summary>
        internal static SearchIndex GetIndex(string name)
        {
            string vectorSearchConfigName = "my-vector-config";

            SearchIndex searchIndex = new(name)
            {
                Fields =
                {
                    new SimpleField("Id", SearchFieldDataType.String) { IsKey = true, IsFilterable = true, IsSortable = true, IsFacetable = true },
                    new SearchableField("Content") { IsFilterable = true },
                    new SearchField("ContentVector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
                    {
                        IsSearchable = true,
                        VectorSearchDimensions = ModelDimensions,
                        VectorSearchConfiguration = vectorSearchConfigName
                    },
                    new SearchableField("Source") { IsFilterable = true, IsSortable = true, IsFacetable = true }
                },
                VectorSearch = new()
                {
                    AlgorithmConfigurations =
                    {
                        new HnswVectorSearchAlgorithmConfiguration(vectorSearchConfigName)
                    }
                },
                //SemanticSettings = new()
                //{
                //    Configurations =
                //    {
                //        new SemanticConfiguration("my-semantic-config", new()
                //        {
                //            TitleField = new(){ FieldName = "Source" },
                //            ContentFields =
                //            {
                //                new() { FieldName = "Content" }
                //            }
                //        })
                //    }
                //}
            };

            return searchIndex;
        }

        internal static void UploadChunks(string path)
        {
            SearchClient searchClient = new(searchEndpoint, VectorSearchIndexName, searchCredential);
            OpenAIClient openAIClient = new(openAIEndpoint, openAICredential);

            List<AzureSDKDocument> docs = new();
            foreach (string line in File.ReadLines(path))
            {
                using (JsonDocument jsonDoc = JsonDocument.Parse(line))
                {
                    JsonElement root = jsonDoc.RootElement;
                    if (root.TryGetProperty("id", out JsonElement id) &&
                        root.TryGetProperty("content", out JsonElement content) &&
                        root.TryGetProperty("source", out JsonElement source))
                    {
                        docs.Add(
                            new AzureSDKDocument()
                            {
                                Id = id.GetString(),
                                Content = content.GetString(),
                                ContentVector = Vectorize(openAIClient, content.GetString()),
                                Source = source.GetString()
                            });
                    }
                }
            }

            // Upload all our docs to Search
            using SearchIndexingBufferedSender<AzureSDKDocument> sender = new(searchClient);
            sender.MergeOrUploadDocuments(docs);
        }

        internal static IReadOnlyList<float> Vectorize(OpenAIClient openAIClient, string text)
        {
            EmbeddingsOptions embeddingsOptions = new(text);
            Embeddings embeddings = openAIClient.GetEmbeddings(ModelName, embeddingsOptions);

            return embeddings.Data[0].Embedding;
        }

        internal class AzureSDKDocument
        {
            public string Id { get; set; }
            public string Content { get; set; }
            public IReadOnlyList<float> ContentVector { get; set; }
            public string Source { get; set; }

            public override bool Equals(object obj) =>
                obj is AzureSDKDocument other &&
                Id == other.Id &&
                Content == other.Content &&
                ContentVector == other.ContentVector &&
                Source == other.Source;

            public override int GetHashCode() => Id?.GetHashCode() ?? 0;
        }
    }
}
