using System;
using System.Collections.Generic;

namespace DijkstraAlgorithm
{
    class Program
    {
        static void Main(string[] args)
        {
            var graph = new Dictionary<char, Dictionary<char, int>> {
                {'A', new Dictionary<char, int> {{'B', 5}, {'C', 1}}},
                {'B', new Dictionary<char, int> {{'A', 5}, {'C', 2}, {'D', 1}}},
                {'C', new Dictionary<char, int> {{'A', 1}, {'B', 2}, {'D', 4}, {'E', 8}}},
                {'D', new Dictionary<char, int> {{'B', 1}, {'C', 4}, {'E', 3}, {'F', 6}}},
                {'E', new Dictionary<char, int> {{'C', 8}, {'D', 3}}},
                {'F', new Dictionary<char, int> {{'D', 6}}}
            };

            var distances = Dijkstra(graph, 'A');

            foreach (var distance in distances)
            {
                Console.WriteLine($"Distance from start to {distance.Key} is {distance.Value}");
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
}

