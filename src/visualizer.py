import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from graph_engine import FraudGraphEngine

def visualize_fraud_network(engine, scores, seed_nodes, output_file="fraud_network.png"):
    """
    Visualizes the graph with nodes colored by suspicion score.
    
    Args:
        engine: The FraudGraphEngine instance (containing the adjacency matrix).
        scores: Dictionary of {node_id: score}.
        seed_nodes: List of seed node IDs (to highlight them).
    """
    print(f"[System] Generating visualization for {engine.num_nodes} nodes...")
    
    # 1. Reconstruct NetworkX Graph from the Engine's adjacency matrix
    G = nx.DiGraph()
    rows, cols = engine.adj_matrix.nonzero()
    for r, c in zip(rows, cols):
        u = engine.reverse_map[r]
        v = engine.reverse_map[c]
        G.add_edge(u, v)
        
    # 2. Setup Node Colors based on Scores
    node_list = list(G.nodes())
    score_values = [scores.get(n, 0.0) for n in node_list]
    
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=min(score_values), vmax=max(score_values))
    
    # 3. Draw with Explicit Axes
    # FIX: Use subplots to get the specific axes object 'ax'
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=node_list,
                           node_color=score_values, 
                           cmap=cmap, 
                           node_size=1000, 
                           edgecolors='black',
                           ax=ax) # Explicitly pass ax
    
    # Highlight Seed Nodes
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=seed_nodes, 
                           node_color='none', 
                           edgecolors='blue', 
                           linewidths=3, 
                           node_size=1200,
                           ax=ax) # Explicitly pass ax

    # Draw Edges and Labels
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Dummy array for the scalar mappable
    
    # FIX: Tell colorbar to steal space from 'ax'
    plt.colorbar(sm, label='Suspicion Score (PPR)', ax=ax)
    
    ax.set_title(f"Fraud Detection Network\nSeeds: {seed_nodes}")
    ax.axis('off')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[System] Visualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    engine = FraudGraphEngine()
    engine.load_graph_from_edgelist("test_graph.txt")
    seeds = ["Fraud1"]
    scores = engine.run_personalized_pagerank(seeds)
    visualize_fraud_network(engine, scores, seeds)