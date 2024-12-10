using NN2;
using System;
using System.Collections.Generic;
using System.Threading;

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
        private int outputSize = 5;
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
            {
                network.InputLayer[i] = rand.NextDouble();
            }
            for (int i = 0; i < network.Target.Length; i++)
            {
                network.Target[i] = rand.NextDouble() * 10;
            }
        }

        private void TrainNetwork()
        {
            for (int i = 0; i < 100; i++)
            {
                network.Feedforward();
                network.Backpropagation();
            }
        }

        private void ConstructGraph()
        {
            graph = new Dictionary<char, Dictionary<char, int>>
    {
        {'A', new Dictionary<char, int> {{'B', GetScaledWeight(0)}, {'C', GetScaledWeight(1)}}},
        {'B', new Dictionary<char, int> {{'A', GetScaledWeight(0)}, {'C', GetScaledWeight(2)}, {'D', GetScaledWeight(3)}}},
        {'C', new Dictionary<char, int> {{'A', GetScaledWeight(1)}, {'B', GetScaledWeight(2)}, {'D', 4}, {'E', 8}}},
        {'D', new Dictionary<char, int> {{'B', GetScaledWeight(3)}, {'C', 4}, {'E', GetScaledWeight(4)}, {'F', 6}}},
        {'E', new Dictionary<char, int> {{'C', 8}, {'D', GetScaledWeight(4)}}},
        {'F', new Dictionary<char, int> {{'D', 6}}}
    };

            // Log all nodes and their connections to check if any are missing
            foreach (var node in graph)
            {
                Console.WriteLine($"Node {node.Key} connects to:");
                foreach (var edge in node.Value)
                {
                    Console.WriteLine($" - {edge.Key} with weight {edge.Value}");
                }
            }
        }

        private int GetScaledWeight(int index)
        {
            return Math.Max(1, (int)(network.OutputLayer[index] * scale));
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

            distances[start] = 0;
            priorityQueue.Add((0, start));

            while (priorityQueue.Count != 0)
            {
                var u = priorityQueue.Min;
                priorityQueue.Remove(u);

                if (!visited.Contains(u.Item2))
                {
                    visited.Add(u.Item2);
                    if (!graph.ContainsKey(u.Item2) || graph[u.Item2] == null)
                    {
                        continue;
                    }
                    foreach (var neighbor in graph[u.Item2])
                    {
                        char v = neighbor.Key;
                        if (!graph[u.Item2].ContainsKey(v))
                        {
                            Console.WriteLine($"Missing edge from {u.Item2} to {v}");
                            continue; // Skip if the edge does not exist
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
