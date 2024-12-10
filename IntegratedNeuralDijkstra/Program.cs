using NN2;
using System;
using System.Collections.Generic;

namespace IntegratedNeuralDijkstra
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetworkDijkstraIntegration integration = new NeuralNetworkDijkstraIntegration();
            var distances = integration.RunDijkstra('A');

            Console.WriteLine("Shortest paths from A:");
            foreach (var distance in distances)
            {
                Console.WriteLine($"To {distance.Key}: {distance.Value}");
            }
        }
    }

    public class NeuralNetworkDijkstraIntegration
    {
        private NeuralNetwork network;
        private Dictionary<char, Dictionary<char, int>> graph;
        private int inputSize = 10;
        private int hiddenSize = 100;
        private int outputSize = 5; // Ideally, this would be linked to the number of edges dynamically
        private int scale = 10; // Scale factor for neural network outputs

        public NeuralNetworkDijkstraIntegration()
        {
            network = new NeuralNetwork(inputSize, hiddenSize, outputSize);
            InitializeNetwork();
            TrainNetwork();
            ConstructGraph();
        }

        private void InitializeNetwork()
        {
            Random rand = new Random();
            for (int i = 0; i < network.InputLayer.Length; i++)
                network.InputLayer[i] = rand.NextDouble();
            for (int i = 0; i < network.Target.Length; i++)
                network.Target[i] = rand.NextDouble() * 10;
        }

        private void TrainNetwork()
        {
            for (int i = 0; i < 100; i++)
            {
                network.Feedforward();
                network.Backpropagation();
            }
        }
        private int GetScaledWeight(int index)
        {
            // Use modulus to cycle through output weights if there are more edges than outputs
            index = index % network.OutputLayer.Length; // Ensure index is within the bounds of available outputs
            return Math.Max(1, (int)(network.OutputLayer[index] * scale));
        }

        private void ConstructGraph()
        {
            graph = new Dictionary<char, Dictionary<char, int>>
    {
        {'A', new Dictionary<char, int> {{'B', GetScaledWeight(0)}, {'C', GetScaledWeight(1)}, {'D', GetScaledWeight(4 % 5)}}}, // Using modular arithmetic
        {'B', new Dictionary<char, int> {{'A', GetScaledWeight(0)}, {'C', GetScaledWeight(2)}, {'E', GetScaledWeight(3)}}},
        {'C', new Dictionary<char, int> {{'A', GetScaledWeight(1)}, {'B', GetScaledWeight(2 % 5)}, {'F', GetScaledWeight(5 % 5)}}}, // Example of cycling
        {'D', new Dictionary<char, int> {{'A', GetScaledWeight(4)}, {'E', GetScaledWeight(3)}, {'G', GetScaledWeight(2)}}},
        {'E', new Dictionary<char, int> {{'B', GetScaledWeight(3)}, {'D', GetScaledWeight(4)}, {'H', GetScaledWeight(1)}}},
        {'F', new Dictionary<char, int> {{'C', GetScaledWeight(5 % 5)}, {'I', GetScaledWeight(4 % 5)}}},
        {'G', new Dictionary<char, int> {{'D', GetScaledWeight(2)}, {'J', GetScaledWeight(1 % 5)}}},
        {'H', new Dictionary<char, int> {{'E', GetScaledWeight(1)}, {'K', GetScaledWeight(2 % 5)}}},
        {'I', new Dictionary<char, int> {{'F', GetScaledWeight(4 % 5)}}},
        {'J', new Dictionary<char, int> {{'G', GetScaledWeight(1 % 5)}}},
        {'K', new Dictionary<char, int> {{'H', GetScaledWeight(2 % 5)}}}
    };
        }



        public Dictionary<char, int> RunDijkstra(char start)
        {
            if (!graph.ContainsKey(start))
            {
                Console.WriteLine($"Start node {start} is not present in the graph.");
                return new Dictionary<char, int>();
            }

            var priorityQueue = new SortedSet<(int, char)>();
            var distances = new Dictionary<char, int>();
            var visited = new HashSet<char>();

            // Initialize distances for all vertices in the graph to prevent KeyNotFoundException
            foreach (var vertex in graph.Keys)
            {
                distances[vertex] = int.MaxValue;
            }
            distances[start] = 0;
            priorityQueue.Add((0, start));

            while (priorityQueue.Count != 0)
            {
                var u = priorityQueue.Min;
                priorityQueue.Remove(u);

                if (!visited.Contains(u.Item2))
                {
                    visited.Add(u.Item2);
                    foreach (var neighbor in graph[u.Item2])
                    {
                        char v = neighbor.Key;
                        if (!distances.ContainsKey(v)) // Additional check for safety
                        {
                            Console.WriteLine($"No distance record for vertex {v}, skipping...");
                            continue;
                        }
                        int weight = graph[u.Item2][v];

                        if (!visited.Contains(v) && distances[u.Item2] + weight < distances[v])
                        {
                            distances[v] = distances[u.Item2] + weight;
                            priorityQueue.Add((distances[v], v));
                        }
                    }
                }
            }

            return distances;
        }

    }
}
