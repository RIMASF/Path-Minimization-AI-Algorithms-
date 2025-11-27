# 📉 Path Minimization Algorithms Project

A Python project that computes the minimum-cost path in an N×N matrix using three different algorithms: Brute Force, Graph-Based (Bellman-Ford), and Dynamic Programming. The goal is to compare correctness and performance across multiple algorithmic strategies.

## ✨ Features
- **Random Matrix Generation** – Creates an N×N grid with values from −1000 to 1000
- **Brute Force Search** – Explores all possible source-to-destination paths
- **Graph-Based Solution** – Uses Bellman-Ford to compute shortest path over graph representation
- **Dynamic Programming** – Efficient minimum-path computation with optimal substructure
- **Execution Time Measurement** – Compares performance across the three approaches

## 🛠️ Technologies
- Python
- NumPy
- Recursion (DFS)
- Bellman-Ford Algorithm
- Dynamic Programming

## 📁 Files
- `AlgorithmsProject2024.py` — Full implementation of all algorithms

## 🚀 How It Works

### Brute Force Algorithm
1. Enumerates all possible paths (up, down, left, right)
2. Records full path costs
3. Selects the least-cost path

### Graph-Based Algorithm (Bellman-Ford)
1. Converts the matrix into a graph of V = N² nodes
2. Adds edges between adjacent cells
3. Computes minimum distance from start to end

### Dynamic Programming
1. Builds a DP table of cumulative minimum costs
2. Fills the table row-wise and column-wise
3. Returns the minimum path sum and reconstructs the optimal path

## 📊 Outputs
- Full path listings (Brute Force)
- Minimum path value using Bellman-Ford
- Minimum path sum using Dynamic Programming
- Execution time for each algorithm

## 🎯 Algorithm Comparison
| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| Brute Force | O(2^(N²)) | O(N) | Small matrices only |
| Bellman-Ford | O(N⁴) | O(N²) | Medium matrices |
| Dynamic Programming | O(N²) | O(N²) | Large matrices |

## 📈 Performance Insights
- **DP is fastest** for large N due to polynomial time
- **Brute Force** only feasible for N ≤ 4
- **Bellman-Ford** serves as intermediate verification method

---

**Status**: ✅ Complete with full implementation and performance analysis  
**Applications**: Route optimization, game pathfinding, network routing
