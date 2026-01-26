# Fraud Detection via Personalized PageRank

**Course:** Data Structures (Fall 2025)  
**Institution:** Shahid Beheshti University  
**Instructor:** Dr. Katanforoush  

## 📌 Project Overview
This project implements a robust, scalable fraud detection engine based on the **"Guilt by Association"** model. Using graph-theoretic analysis—specifically **Personalized PageRank (PPR)**—the system propagates a "suspicion score" from a seed set of known fraudulent entities to the rest of the network.

The core engine is built on **Sparse Matrix** logic (Compressed Sparse Row) to ensure $O(V+E)$ memory complexity, allowing it to handle large-scale real-world graphs efficiently.

## 🚀 Features
* **Custom PPR Engine:** Implements the Power Iteration method: $r^{(t+1)} = (1-\alpha)r^{(t)}M + \alpha p$.
* **Sparse Optimization:** Uses `scipy.sparse` for efficient memory usage on large graphs.
* **Edge Case Handling:**
    * **Rank Sinks:** Automatically redistributes mass from dangling nodes (dead ends) to maintain stochasticity.
    * **Disconnected Components:** Handles graph fragmentation gracefully.
* **Scientific Visualization:** Generates heatmaps of suspicion propagation.
* **Analysis Suite:** Includes modules for linear scalability testing and parameter sensitivity analysis.
* **Bonus:** Includes a Monte Carlo Random Walk approximation to compare accuracy vs. speed against the exact matrix method.

## 📂 Repository Structure
```text
.
├── data/                   # Dataset storage (e.g., Caltech Facebook network)
├── docs/                   # Final Report PDF
├── results/                # Generated plots and visualization images
├── src/
│   ├── graph_engine.py     # Core PPR implementation (Sparse Matrix Engine)
│   ├── visualizer.py       # Graph heatmap generation script
│   ├── scalability_test.py # O(E) runtime benchmark script
│   ├── sensitivity_analysis.py # Damping factor analysis script
│   └── bonus_test.py       # Monte Carlo vs Power Iteration comparison
└── README.md

```

## 🛠️ Prerequisites
The system relies on scientific computing libraries for matrix algebra and plotting.

* Python 3.10+

* Dependencies:
```text
Bash
pip install numpy scipy networkx matplotlib requests
```
## 💻 Usage
### 1. Visualizing Fraud Propagation

Runs the algorithm on a synthetic graph to demonstrate how suspicion accumulates in cyclic structures (traps).
```text
Bash
python src/visualizer.py
```
Output: Generates results/fraud_network.png.

### 2. Scalability Benchmark

Downloads the Caltech Facebook dataset (~4k nodes, ~88k edges) and measures runtime across growing graph sizes to verify linear complexity.
```text
Bash
python src/scalability_test.py
```
Output: Generates results/scalability_plot_fixed.png.

### 3. Parameter Sensitivity Analysis

Analyzes how the Damping Factor (α) impacts the depth of suspicion spread.
```text
Bash
python src/sensitivity_analysis.py
```
Output: Generates results/sensitivity_plot.png.

### 4. Monte Carlo Approximation (Bonus)

Compares the accuracy and execution time of the Exact Power Iteration method against a Monte Carlo Random Walk simulation.
```text
Bash
python src/bonus_test.py
```
Output: Generates results/bonus_monte_carlo.png.

## ⚙️ Configuration
### Key parameters in src/graph_engine.py:

* damping_factor (default 0.85): Probability of continuing the walk (1−α).

* epsilon (default 1e-6): Convergence threshold for the L 
1
​	
  norm.

* max_iterations (default 100): Safety stop for power iteration.


## 📜 References
* Page, L., et al. (1999). The PageRank Citation Ranking: Bringing Order to the Web.

* Gyöngyi, Z., et al. (2004). Combating Web Spam with TrustRank.
