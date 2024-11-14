// using System;
// using Npgsql;

// class InsertVectorToPostgres
// {
//     static void Main(string[] args)
//     {
//         // Connection string for your PostgreSQL database
//         var connectionString = "Host=localhost;Username=postgres;Password=test;Database=postgres";

//         // Sample description and embedding vector
//         string description = "This is a sample text";
//         float[] embedding = new float[] { 0.1f, 0.2f, 0.3f, /* ... */ 0.768f }; // Replace with your actual vector

//         using var connection = new NpgsqlConnection(connectionString);
//         connection.Open();

//         // Prepare the INSERT statement
//         using var command = new NpgsqlCommand("INSERT INTO embeddings (description, embedding) VALUES (@description, @embedding)", connection);
//         command.Parameters.AddWithValue("description", description);
//         command.Parameters.AddWithValue("embedding", embedding);

//         // Execute the command
//         command.ExecuteNonQuery();

//         Console.WriteLine("Embedding inserted successfully!");
//     }
// }