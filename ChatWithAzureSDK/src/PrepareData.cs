using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using HtmlAgilityPack;

namespace ChatWithAzureSDK
{
    public class PrepareData
    {
        public static List<ExtractedDocument> ParseHtmlContent()
        {
            string directoryPath = @"C:\Users\shreja\Demo\ChatWithAzureSDK\ChatWithAzureSDK\src\testdocs";
            List<ExtractedDocument> documents = new List<ExtractedDocument>();
            ProcessDirectory(directoryPath, documents);
            return documents;
        }

        public static void ProcessDirectory(string directoryPath, List<ExtractedDocument> documents)
        {
            foreach (string filePath in Directory.GetFiles(directoryPath, "*.html"))
            {
                string fileContent = File.ReadAllText(filePath, Encoding.UTF8);
                HtmlDocument document = new HtmlDocument();
                document.LoadHtml(fileContent);

                StringBuilder extractedTexts = new StringBuilder();

                foreach (HtmlNode node in document.DocumentNode.DescendantsAndSelf())
                {
                    string text = node.InnerText.Trim();
                    if (!string.IsNullOrEmpty(text))
                    {
                        extractedTexts.Append(text);
                    }
                }

                ExtractedDocument dataInstance = new ExtractedDocument { FilePath = filePath, PageContent = extractedTexts.ToString() };
                documents.Add(dataInstance);
            }

            foreach (string subdirectory in Directory.GetDirectories(directoryPath))
            {
                ProcessDirectory(subdirectory, documents);
            }
        }

        // Custom class to hold extracted data
        public class ExtractedDocument
        {
            public string FilePath { get; set; }
            public string PageContent { get; set; }
        }
    }
}
