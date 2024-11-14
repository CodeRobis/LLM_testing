using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Npgsql;

class Program
{
     static void Main(string[] args)
    {
        // Paths to the model and tokenizer
        string modelPath = @"C:/Users/omegauser/onnx_model/model.onnx";
        string tokenizerPath = @"C:/Users/omegauser/Desktop/Omega/LLM_testing/tokenizer/tokenizer.json";

        try
        {
            // Initialize the inference session
            using var session = new InferenceSession(modelPath);
            Console.WriteLine("ONNX model loaded successfully.");

            // Load the tokenizer configuration
            var tokenizerJson = File.ReadAllText(tokenizerPath);
            var tokenizerConfig = JsonConvert.DeserializeObject<TokenizerConfig>(tokenizerJson);

            // Check if the vocabulary is loaded correctly
            if (tokenizerConfig?.Model?.Vocab == null)
            {
                Console.WriteLine("Error: Vocabulary is null. Please check the tokenizer configuration.");
                return;
            }

            var vocabulary = tokenizerConfig.Model.Vocab;
            //string inputText = "The AST_QualityMgmt_R4_ITPRegister_SetContract test is a regression test designed to verify that the Set Contract feature in the ITP Register module of the Quality Management domain works correctly. It involves logging into the application using different sets of credentials, navigating to the ITP Register section, and attempting to add a contract. The test asserts that the operation is successful and, if any issues occur, captures a screenshot and logs the exception for further analysis. Finally, the test performs a cleanup step to roll back any changes made during the test, ensuring no residual impact on the system state.";
            //string inputText = "The AST_QualityMgmt_R4_DomainSetup_EnableFindingHeatmap test is a regression test aimed at verifying that the Enable Finding Heatmap feature within the Domain Setup module of the Quality Management application functions as intended. The test involves logging into the application using provided credentials, navigating to the Domain Setup section, and enabling the Finding Heatmap feature. After saving the changes, it navigates to a specific finding within the Activity Details module and checks whether the Finding Heatmap is visible. The test asserts that the heatmap is correctly displayed, and if any issues occur, it captures a screenshot and logs the exception for further investigation.";
            string inputText = "The AST_QualityMgmt_R4_ActivityDetails_AddGeoLocation test is a regression test that verifies the ability to add a geographic location (latitude and longitude) to an activity within the Activity Details module of the Quality Management domain. The test logs into the application using the provided host, username, and password, then navigates to the specified activity within the domain. It sets the geographic coordinates to a predefined latitude and longitude and asserts that the operation is successful. If any issues occur, the test captures a screenshot and logs the exception for further investigation. Finally, a cleanup step is performed to roll back any changes made during the test, ensuring no residual impact on the system state.";
            // Tokenize the input text
            var encoding = CustomTokenizer.Encode(inputText, vocabulary);

            // Convert tokens to tensors with long[]
            var inputIds = encoding.Ids.Select(id => (long)id).ToArray();
            var attentionMask = encoding.AttentionMask.Select(mask => (long)mask).ToArray();
            var tokenTypeIds = new long[inputIds.Length]; // All zeros for single-segment input

            // Check if input tensors are valid
            if (inputIds.Length == 0 || attentionMask.Length == 0)
            {
                Console.WriteLine("Error: Tokenization resulted in empty input tensors.");
                return;
            }

            // Create input tensors
            var inputIdsTensor = new DenseTensor<long>(inputIds, new int[] { 1, inputIds.Length });
            var attentionMaskTensor = new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length });
            var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, new int[] { 1, tokenTypeIds.Length });

            // Prepare input data
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
            };

            // Run inference
            using var results = session.Run(inputs);

            // Extract the output tensor (assumes the output contains embeddings for each token)
            var output = results.First().AsTensor<float>();
            var tokenEmbeddings = output.ToArray();

            // Apply mean pooling to get a single 384-dimensional vector
            int embeddingSize = 384; // Adjust if your model's output size is different
            var pooledEmbedding = new float[embeddingSize];

            for (int i = 0; i < output.Dimensions[1]; i++) // Loop over each token's embedding
            {
                for (int j = 0; j < embeddingSize; j++)
                {
                    pooledEmbedding[j] += tokenEmbeddings[i * embeddingSize + j];
                }
            }

            // Average the pooled values
            for (int j = 0; j < embeddingSize; j++)
            {
                pooledEmbedding[j] /= output.Dimensions[1];
            }

            Console.WriteLine($"Embedding dimension: {pooledEmbedding.Length}");

            // Connection string for your PostgreSQL database
            var connectionString = "Host=localhost;Username=postgres;Password=test;Database=postgres";

            using var connection = new NpgsqlConnection(connectionString);
            connection.Open();

            // Prepare the INSERT statement
            using var command = new NpgsqlCommand("INSERT INTO embeddings (description, embedding) VALUES (@description, @embedding)", connection);
            command.Parameters.AddWithValue("description", inputText);
            command.Parameters.AddWithValue("embedding", pooledEmbedding);

            // Execute the command
            command.ExecuteNonQuery();

            Console.WriteLine("Embedding inserted successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }
}

// Custom classes for parsing and tokenization
public class TokenizerConfig
{
    [JsonProperty("model")]
    public ModelConfig Model { get; set; }
}

public class ModelConfig
{
    [JsonProperty("vocab")]
    public Dictionary<string, int> Vocab { get; set; }
}

public static class CustomTokenizer
{
    public static EncodingResult Encode(string text, Dictionary<string, int> vocab)
    {
        if (vocab == null)
        {
            throw new ArgumentNullException(nameof(vocab), "Vocabulary dictionary cannot be null.");
        }

        // Implement your tokenization logic here
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var ids = words.Select(word => vocab.ContainsKey(word) ? vocab[word] : (vocab.ContainsKey("[UNK]") ? vocab["[UNK]"] : -1)).ToArray();

        // Check for any -1 values, indicating unknown words that couldn't be mapped
        if (ids.Contains(-1))
        {
            Console.WriteLine("Warning: Some words could not be mapped to a valid ID.");
        }

        var attentionMask = ids.Select(_ => 1L).ToArray(); // Note the use of long (1L) for attention mask

        return new EncodingResult
        {
            Ids = ids.Select(id => (long)id).ToArray(), // Convert ids to long[]
            AttentionMask = attentionMask
        };
    }
}

public class EncodingResult
{
    public long[] Ids { get; set; }
    public long[] AttentionMask { get; set; }
}
