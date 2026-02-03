import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict, Set
import time

class FraudGraphEngine:
    """
    A robust engine for Fraud Detection using Personalized PageRank (PPR).
    Implements Power Iteration on Compressed Sparse Row (CSR) matrices.
    """

    def __init__(self, damping_factor: float = 0.85, epsilon: float = 1.0e-6, max_iterations: int = 100):
        """
        Initialize the engine parameters.
        
        Args:
            damping_factor (alpha): Probability of following a link vs teleporting (usually 0.85).
                                    Note: In your math, alpha is teleportation prob (0.15).
                                    We use standard convention: damping = 1 - alpha.
            epsilon: Convergence threshold for L1 norm.
            max_iterations: Safety break to prevent infinite loops.
        """
        self.damping = damping_factor
        self.teleport_prob = 1.0 - damping_factor  # This is the 'alpha' in your prompt [cite: 25]
        self.epsilon = epsilon
        self.max_iter = max_iterations
        
        # Graph State
        self.adj_matrix = None  # The raw adjacency matrix
        self.transition_matrix = None  # Row-normalized matrix M
        self.node_map = {}  # Map real IDs to matrix indices
        self.reverse_map = {} # Map matrix indices to real IDs
        self.num_nodes = 0
        self.dangling_nodes = None # Boolean mask for dead ends

    def load_graph_from_edgelist(self, file_path: str, weighted: bool = False):
        """
        Parses an edge list file and builds the Sparse Transition Matrix M.
        Supports weighted and unweighted directed graphs[cite: 31].
        
        Expected Format: SourceID TargetID [Weight]
        """
        print(f"[System] Loading graph from {file_path}...")
        sources = []
        targets = []
        weights = []
        
        # 1. First Pass: Map unique IDs to contiguous integers 0..N-1
        unique_nodes = set()
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 2: continue
                
                u_id, v_id = parts[0], parts[1]
                w = float(parts[2]) if weighted and len(parts) > 2 else 1.0
                
                unique_nodes.add(u_id)
                unique_nodes.add(v_id)
                
                sources.append(u_id)
                targets.append(v_id)
                weights.append(w)

        # Create mappings
        self.node_map = {uid: i for i, uid in enumerate(sorted(unique_nodes))}
        self.reverse_map = {i: uid for uid, i in self.node_map.items()}
        self.num_nodes = len(unique_nodes)
        
        # Convert raw IDs to internal Matrix Indices
        src_indices = [self.node_map[u] for u in sources]
        tgt_indices = [self.node_map[v] for v in targets]
        
        # 2. Build Compressed Sparse Row (CSR) Matrix
        # This satisfies the "Sparse Matrix Optimization" requirement [cite: 26]
        self.adj_matrix = sparse.csr_matrix(
            (weights, (src_indices, tgt_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        # 3. Construct Transition Matrix M
        # Normalize rows so they sum to 1
        degree = np.array(self.adj_matrix.sum(axis=1)).flatten()
        
        # Detect Dangling Nodes (Degree 0) 
        self.dangling_nodes = np.where(degree == 0)[0]
        
        # Avoid division by zero for dangling nodes
        degree[degree == 0] = 1.0
        
        # D_inv = diag(1/degree)
        D_inv = sparse.diags(1.0 / degree)
        
        # M = D_inv * A
        self.transition_matrix = D_inv.dot(self.adj_matrix)
        
        print(f"[System] Graph built: {self.num_nodes} nodes, {len(weights)} edges.")
        print(f"[System] Detected {len(self.dangling_nodes)} dangling nodes (dead ends).")

    def run_personalized_pagerank(self, seed_ids: List[str]) -> Dict[str, float]:
        """
        Executes the Custom PageRank Engine using Power Iteration[cite: 24].
        
        Args:
            seed_ids: List of node IDs known to be fraudulent (The Seed Set).
        """
        if self.transition_matrix is None:
            raise ValueError("Graph not loaded. Call load_graph_from_edgelist first.")

        # 1. Create Personalization Vector (p)
        # Non-zero only for known fraudsters [cite: 25]
        p = np.zeros(self.num_nodes)
        valid_seeds = [self.node_map[s] for s in seed_ids if s in self.node_map]
        
        if not valid_seeds:
            raise ValueError("No valid seed nodes found in the graph.")
            
        # Normalize p so it sums to 1
        p[valid_seeds] = 1.0 / len(valid_seeds)
        
        # 2. Initialize Rank Vector (r)
        # Start with the personalization vector distribution
        r = p.copy()
        
        print(f"[System] Starting Power Iteration with alpha={self.teleport_prob}...")
        
        # 3. Power Iteration Loop [cite: 24]
        for iteration in range(self.max_iter):
            r_prev = r.copy()
            
            # Matrix Multiplication: r_new = damping * (r * M)
            # This spreads suspicion to neighbors
            r_new = self.damping * (r @ self.transition_matrix)
            
            # Handling Dangling Nodes [cite: 29]
            # Calculate mass lost in dead ends: sum(r[dangling_nodes])
            # This mass is redistributed according to p (teleport back to seeds)
            dangling_mass = np.sum(r[self.dangling_nodes])
            r_new += (self.damping * dangling_mass) * p
            
            # Teleportation Step (The Personalization)
            # Add the restart probability mass (alpha * p)
            r_new += self.teleport_prob * p
            
            # 4. Convergence Criteria 
            # L1 Norm Check: ||r_new - r_prev||_1
            error = np.linalg.norm(r_new - r_prev, 1)
            
            if error < self.epsilon:
                print(f"[System] Converged at iteration {iteration+1}. Error: {error:.2e}")
                break
            
            r = r_new
        else:
            print(f"[Warning] Reached max iterations ({self.max_iter}) without full convergence.")

        # Map back to string IDs and return
        scores = {self.reverse_map[i]: s for i, s in enumerate(r)}
        return scores

    def run_monte_carlo_ppr(self, seed_ids: List[str], num_walks: int = 10000) -> Dict[str, float]:
        """
        Bonus Task: Approximates PPR using Monte Carlo Random Walks.
        Instead of matrix multiplication, we simulate thousands of 'drunk' agents.
        
        Algorithm:
        1. Start 'num_walks' particles at the seed nodes.
        2. In each step, a particle has 'teleport_prob' chance to stop.
        3. If it doesn't stop, it moves to a random neighbor.
        4. The final distribution of particles is the approximate PageRank.
        """
        import random
        
        if self.adj_matrix is None:
            raise ValueError("Graph not loaded.")

        # Convert seeds to internal indices
        valid_seeds = [self.node_map[s] for s in seed_ids if s in self.node_map]
        if not valid_seeds:
            raise ValueError("No valid seeds.")
            
        # Track where particles land
        visit_counts = np.zeros(self.num_nodes, dtype=int)
        
        print(f"[System] Running Monte Carlo Approximation ({num_walks} walks)...")
        
        # We need efficient lookup for neighbors
        # Convert CSR to list of lists for faster random access in Python loop
        # (Or stay in CSR and use index slicing, but list is easier for logic)
        adj_list = []
        for i in range(self.num_nodes):
            # Get row slice
            row_start = self.adj_matrix.indptr[i]
            row_end = self.adj_matrix.indptr[i+1]
            neighbors = self.adj_matrix.indices[row_start:row_end]
            adj_list.append(neighbors)

        # Simulation Loop
        for _ in range(num_walks):
            # 1. Pick a random start from seeds
            curr_node = random.choice(valid_seeds)
            
            # 2. Walk until we 'teleport' (stop)
            while True:
                # Coin flip: Do we stop?
                if random.random() < self.teleport_prob:
                    # STOP and record position
                    visit_counts[curr_node] += 1
                    break
                
                # Move to neighbor
                neighbors = adj_list[curr_node]
                
                if len(neighbors) == 0:
                    # Dead end: In PPR logic, we teleport back to seed
                    curr_node = random.choice(valid_seeds)
                    # (Or we could stop here. But restarting is standard for dead ends)
                else:
                    curr_node = random.choice(neighbors)
        
        # Normalize counts to get probabilities
        probabilities = visit_counts / num_walks
        
        # Map back to string IDs (return top non-zero only to save space)
        scores = {self.reverse_map[i]: p for i, p in enumerate(probabilities) if p > 0}
        return scores

# --- Usage Example ---
if __name__ == "__main__":
    # Create a dummy edge list file for testing
    with open("test_graph.txt", "w") as f:
        f.write("UserA UserB\n") # A trusts B
        f.write("UserB UserC\n") # B trusts C
        f.write("UserC UserA\n") # C trusts A
        f.write("UserD UserA\n") # D trusts A
        f.write("Fraud1 UserB\n") # Fraudster interacts with B
        f.write("Fraud1 Fraud2\n") # Fraudster interacts with another Fraudster
    
    # Initialize Engine
    engine = FraudGraphEngine(damping_factor=0.85, epsilon=1e-6)
    
    # Load Data
    engine.load_graph_from_edgelist("test_graph.txt")
    
    # Run Fraud Detection
    # Seed set includes 'Fraud1'
    # We expect 'UserB' to have a higher score than 'UserD' due to proximity
    scores = engine.run_personalized_pagerank(seed_ids=["Fraud1"])
    
    # Sort and Print top suspicious nodes
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Suspicion Scores (Top 5) ---")
    for uid, score in sorted_scores[:5]:
        print(f"Node: {uid}, Score: {score:.6f}")
