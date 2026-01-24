import matplotlib.pyplot as plt
import numpy as np
from graph_engine import FraudGraphEngine

def run_sensitivity_test(graph_path="data/facebook_combined.txt"):
    print("[System] Loading graph for Sensitivity Analysis...")
    
    # Define alphas (damping factors) to test
    # Note: In our engine, 'damping' is the probability of CONTINUING the walk.
    # High damping (0.95) = Long walks (suspicion spreads far).
    # Low damping (0.50) = Short walks (suspicion stays close to seed).
    damping_factors = [0.50, 0.85, 0.95, 0.99]
    
    # Store top 100 scores for each factor to compare distributions
    results = {}
    
    # 1. Initialize and Load
    engine = FraudGraphEngine()
    engine.load_graph_from_edgelist(graph_path)
    
    # Pick the most connected node as seed (stable choice)
    degrees = np.array(engine.adj_matrix.sum(axis=1)).flatten()
    seed_node = engine.reverse_map[np.argmax(degrees)]
    print(f"[System] Using consistent seed: {seed_node}")

    # 2. Run Experiments
    for d in damping_factors:
        print(f"   -> Testing damping factor: {d}")
        # Update engine parameter
        engine.damping = d
        engine.teleport_prob = 1.0 - d
        
        # Run
        scores = engine.run_personalized_pagerank([seed_node])
        
        # Sort scores descending and take top 100 (excluding the seed itself for clearer view)
        sorted_scores = sorted(scores.values(), reverse=True)
        results[d] = sorted_scores[:100]

    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    for d, score_list in results.items():
        plt.plot(score_list, label=f'Damping (1-α) = {d}')
        
    plt.yscale('log') # Log scale helps distinguish the tail behavior
    plt.title("Parameter Sensitivity: Effect of Damping Factor on Suspicion Spread")
    plt.xlabel("Rank of Node (Top 100)")
    plt.ylabel("Suspicion Score (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    output_file = "results/sensitivity_plot.png"
    plt.savefig(output_file)
    print(f"[System] Analysis complete. Plot saved to {output_file}")

if __name__ == "__main__":
    # Ensure you have the facebook data downloaded via the previous script first
    run_sensitivity_test()