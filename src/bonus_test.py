import time
import numpy as np
import matplotlib.pyplot as plt
from graph_engine import FraudGraphEngine

def compare_algorithms():
    print("--- Bonus Task: Monte Carlo vs Power Iteration ---")
    
    # 1. Setup
    engine = FraudGraphEngine()
    engine.load_graph_from_edgelist("data/facebook_combined.txt")
    
    # Pick a high-degree seed
    degrees = np.array(engine.adj_matrix.sum(axis=1)).flatten()
    seed = engine.reverse_map[np.argmax(degrees)]
    
    # 2. Run Exact Method (Power Iteration)
    print("\n[1] Running Exact Power Iteration...")
    t0 = time.time()
    exact_scores = engine.run_personalized_pagerank([seed])
    t_exact = time.time() - t0
    print(f"    Time: {t_exact:.4f}s")

    # 3. Run Approximate Method (Monte Carlo)
    # We try different walk counts to show the trade-off
    walk_counts = [1000, 10000, 100000]
    errors = []
    times = []
    
    # Convert exact scores to vector for comparison
    exact_vec = np.zeros(engine.num_nodes)
    for node, score in exact_scores.items():
        if node in engine.node_map:
            exact_vec[engine.node_map[node]] = score

    print("\n[2] Running Monte Carlo Approximations...")
    for N in walk_counts:
        t0 = time.time()
        mc_scores = engine.run_monte_carlo_ppr([seed], num_walks=N)
        t_mc = time.time() - t0
        
        # Convert to vector
        mc_vec = np.zeros(engine.num_nodes)
        for node, score in mc_scores.items():
            if node in engine.node_map:
                mc_vec[engine.node_map[node]] = score
                
        # Calculate Error (L1 diff from exact)
        error = np.sum(np.abs(exact_vec - mc_vec))
        
        errors.append(error)
        times.append(t_mc)
        print(f"    Walks: {N:<7} | Time: {t_mc:.4f}s | L1 Error: {error:.4f}")

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Random Walks (Log Scale)')
    ax1.set_ylabel('L1 Error (Accuracy)', color=color)
    ax1.plot(walk_counts, errors, marker='o', color=color, label='Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Execution Time (s)', color=color)
    ax2.plot(walk_counts, times, marker='s', linestyle='--', color=color, label='Time')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add exact time reference
    plt.axhline(y=t_exact, color='green', linestyle=':', label='Exact Method Time')

    plt.title("Monte Carlo Approximation: Speed vs Accuracy Trade-off")
    fig.tight_layout()
    plt.savefig("results/bonus_monte_carlo.png")
    print("\n[System] Bonus analysis saved to results/bonus_monte_carlo.png")

if __name__ == "__main__":
    compare_algorithms()