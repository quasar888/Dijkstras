using System;
using System.Collections.Generic;
using System.Threading;
using NN2;

namespace IntegratedNeuralDijkstra
{
    class Program
    {
        static void Main(string[] args)
        {
            // Neural Network setup
            int inputSize = 10;
            int hiddenSize = 100;
            int outputSize = 5;  // This will directly correspond to some graph weights
            NeuralNetwork network = new NeuralNetwork(inputSize, hiddenSize, outputSize);

            // Randomly initialize inputs and targets for demonstration
            Random rand = new Random();
            for (int i = 0; i < network.InputLayer.Length; i++)
            {
                network.InputLayer[i] = rand.NextDouble();  // Random input values
            }
            for (int i = 0; i < network.Target.Length; i++)
            {
                network.Target[i] = rand.NextDouble() * 10;  // Random targets, assuming some path costs
            }

            // Neural network training loop (simplified)
            for (int i = 0; i < 100; i++)  // Run the training loop 100 times
            {
                network.Feedforward();
                network.Backpropagation();
            }

            // Use neural network output to set weights in the graph
            int scale = 10;  // Scale up factor to increase weight impact
            var graph = new Dictionary<char, Dictionary<char, int>>
            {
                {'A', new Dictionary<char, int> {{'B', Math.Max(1, (int)(network.OutputLayer[0] * scale))}, {'C', Math.Max(1, (int)(network.OutputLayer[1] * scale))}}},
                {'B', new Dictionary<char, int> {{'A', Math.Max(1, (int)(network.OutputLayer[0] * scale))}, {'C', Math.Max(1, (int)(network.OutputLayer[2] * scale))}, {'D', Math.Max(1, (int)(network.OutputLayer[3] * scale))}}},
                {'C', new Dictionary<char, int> {{'A', Math.Max(1, (int)(network.OutputLayer[1] * scale))}, {'B', Math.Max(1, (int)(network.OutputLayer[2] * scale))}, {'D', 4}, {'E', 8}}},
                {'D', new Dictionary<char, int> {{'B', Math.Max(1, (int)(network.OutputLayer[3] * scale))}, {'C', 4}, {'E', Math.Max(1, (int)(network.OutputLayer[4] * scale))}, {'F', 6}}},
                {'E', new Dictionary<char, int> {{'C', 8}, {'D', Math.Max(1, (int)(network.OutputLayer[4] * scale))}}},
                {'F', new Dictionary<char, int> {{'D', 6}}}
            };

            // Run Dijkstra's algorithm
            var distances = Dijkstra(graph, 'A');

            // Output the shortest paths
            Console.WriteLine("Shortest paths from A:");
            foreach (var distance in distances)
            {
                Console.WriteLine($"To {distance.Key}: {distance.Value}");
            }
        }

        static Dictionary<char, int> Dijkstra(Dictionary<char, Dictionary<char, int>> graph, char start)
        {
            var priorityQueue = new SortedSet<(int, char)>(Comparer<(int, char)>.Create((a, b) => {
                int compare = a.Item1.CompareTo(b.Item1);
                if (compare == 0) return a.Item2.CompareTo(b.Item2);
                return compare;
            }));
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
                    foreach (var neighbor in graph[u.Item2])
                    {
                        char v = neighbor.Key;
                        int weight = neighbor.Value;

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
            ConstructGraph();
        }

        private void InitializeNetwork()
        {
            Random rand = new Random();
            // Initialize network inputs and targets
            for (int i = 0; i < network.InputLayer.Length; i++)
            {
                network.InputLayer[i] = rand.NextDouble(); // Random input values
            }
            for (int i = 0; i < network.Target.Length; i++)
            {
                network.Target[i] = rand.NextDouble() * 10; // Random target values
            }
        }

        private void ConstructGraph()
        {
            // Assume the network is trained and outputs are available
            network.Feedforward();

            graph = new Dictionary<char, Dictionary<char, int>>
            {
                {'A', new Dictionary<char, int> {{'B', GetScaledWeight(0)}, {'C', GetScaledWeight(1)}}},
                {'B', new Dictionary<char, int> {{'A', GetScaledWeight(0)}, {'C', GetScaledWeight(2)}, {'D', GetScaledWeight(3)}}},
                {'C', new Dictionary<char, int> {{'A', GetScaledWeight(1)}, {'B', GetScaledWeight(2)}, {'D', 4}, {'E', 8}}},
                {'D', new Dictionary<char, int> {{'B', GetScaledWeight(3)}, {'C', 4}, {'E', GetScaledWeight(4)}, {'F', 6}}},
                {'E', new Dictionary<char, int> {{'C', 8}, {'D', GetScaledWeight(4)}}},
                {'F', new Dictionary<char, int> {{'D', 6}}}
            };
        }

        private int GetScaledWeight(int index)
        {
            return Math.Max(1, (int)(network.OutputLayer[index] * scale));
        }

        public Dictionary<char, int> RunDijkstra(char start)
        {
            var priorityQueue = new SortedSet<(int, char)>();
            var distances = new Dictionary<char, int>();
            var visited = new HashSet<char>();

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

                if (visited.Contains(u.Item2))
                {
                    continue;
                }
                visited.Add(u.Item2);

                foreach (var neighbor in graph[u.Item2])
                {
                    char v = neighbor.Key;
                    int weight = neighbor.Value;

                    if (!visited.Contains(v) && distances[u.Item2] + weight < distances[v])
                    {
                        distances[v] = distances[u.Item2] + weight;
                        priorityQueue.Add((distances[v], v));
                    }
                }
            }

            return distances;
        }
    }
}
