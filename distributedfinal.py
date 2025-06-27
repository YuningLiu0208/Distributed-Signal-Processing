

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import networkx as nx
from matplotlib.patches import Circle
import time

np.random.seed(42)

class SensorNetwork:
    """Sensor Network Class"""
    def __init__(self, n_sensors=200, area_size=100, transmission_radius=20):
        self.n_sensors = n_sensors
        self.area_size = area_size
        self.transmission_radius = transmission_radius
        self.positions = None
        self.adjacency = None
        self.dist_matrix = None
        self.graph = None

    def generate_network(self):
        """Generate sensor network"""
        self.positions = np.random.uniform(0, self.area_size, (self.n_sensors, 2))
        self.dist_matrix = distance_matrix(self.positions, self.positions)
        self.adjacency = (self.dist_matrix <= self.transmission_radius) & (self.dist_matrix > 0)
        self.graph = nx.from_numpy_array(self.adjacency.astype(int))
        return self.is_connected()

    def is_connected(self):
        """Check if network is connected"""
        return nx.is_connected(self.graph)

    def get_network_stats(self):
        """Get network statistics"""
        if not self.is_connected():
            return None
        stats = {
            'nodes': self.n_sensors,
            'edges': self.graph.number_of_edges(),
            'average_degree': 2 * self.graph.number_of_edges() / self.n_sensors,
            'diameter': nx.diameter(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph),
            'clustering_coefficient': nx.average_clustering(self.graph)
        }
        return stats

    def visualize_network(self):
        """Visualize sensor network"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-5, self.area_size + 5)
        ax.set_ylim(-5, self.area_size + 5)
        ax.set_aspect('equal')
        ax.scatter(self.positions[:, 0], self.positions[:, 1],
                   c='red', s=50, zorder=5, label='Sensor nodes')
        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                if self.adjacency[i, j]:
                    ax.plot([self.positions[i, 0], self.positions[j, 0]],
                            [self.positions[i, 1], self.positions[j, 1]],
                            'b-', alpha=0.3, linewidth=0.5)
        for i in range(min(5, self.n_sensors)):
            circle = Circle(self.positions[i], self.transmission_radius,
                            fill=False, edgecolor='green', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_title(f'Sensor Network (n={self.n_sensors}, r={self.transmission_radius}m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

class RandomizedGossip:
    """Randomized Gossip Algorithm Class"""
    def __init__(self, adjacency_matrix, initial_values):
        self.n = len(initial_values)
        self.adjacency = adjacency_matrix
        self.values = initial_values.copy()
        self.initial_values = initial_values.copy()
        self.history = [self.values.copy()]
        self.transmissions = 0
        self.transmissions_history = [0]
        self.iterations = 0
        self.true_average = np.mean(initial_values)
        self.neighbors = {}
        for i in range(self.n):
            self.neighbors[i] = np.where(self.adjacency[i])[0]

    def step(self):
        """Execute one Gossip update step"""
        i = np.random.randint(0, self.n)
        if len(self.neighbors[i]) > 0:
            j = np.random.choice(self.neighbors[i])
            avg = (self.values[i] + self.values[j]) / 2
            self.values[i] = avg
            self.values[j] = avg
            self.transmissions += 2
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)

    def run(self, max_iterations=100000, tolerance=1e-8):
        """Run Gossip algorithm until convergence"""
        start_time = time.time()
        for k in range(max_iterations):
            self.step()
            error = np.linalg.norm(self.values - self.true_average) / np.linalg.norm(self.initial_values)
            if error < tolerance:
                elapsed_time = time.time() - start_time
                print(f"Gossip converged at iteration {k+1}")
                print(f"Final error: {error:.2e}")
                print(f"Total transmissions: {self.transmissions}")
                print(f"Runtime: {elapsed_time:.2f} seconds")
                break
            if (k + 1) % 10000 == 0:
                print(f"Gossip iteration {k+1}: error = {error:.2e}")
        return self.history, self.transmissions, self.transmissions_history

    def compute_error_history(self):
        """Compute normalized L2 error history"""
        history_array = np.array(self.history)
        errors = []
        for values in history_array:
            error = np.linalg.norm(values - self.true_average) / np.linalg.norm(self.initial_values)
            errors.append(error)
        return errors

class PDMM:
    """Primal-Dual Method of Multipliers (PDMM) Algorithm Class"""
    def __init__(self, adjacency_matrix, initial_values, c=0.1, mode='average',
                 broadcast=False, transmission_loss=0, active_nodes=1, theta_mode='1/2-avg'):
        self.n = len(initial_values)
        self.initial_values = initial_values.copy()
        self.c = c
        self.mode = mode
        self.broadcast = broadcast
        self.transmission_loss = transmission_loss
        self.active_nodes = active_nodes

        # Set theta based on theta_mode
        if theta_mode == '1/2-avg':
            self.theta = 0.5  # ADMM equivalent
        elif theta_mode == 'std':
            self.theta = 1.0  # Standard PDMM
        else:
            raise ValueError("theta_mode must be '1/2-avg' or 'std'")

        self.theta_mode = theta_mode

        if self.mode == 'average':
            self.true_target = np.mean(initial_values)
        elif self.mode == 'median':
            self.true_target = np.median(initial_values)
        else:
            raise ValueError("Mode must be 'average' or 'median'")

        self.iterations = 0
        self.history = []
        self.transmissions = 0
        self.transmissions_history = []

        self.adj_matrix = adjacency_matrix
        self.degrees = np.sum(self.adj_matrix, axis=1)

        self.values = initial_values.copy()

        if self.mode == 'average':
            self.z = {(i, j): np.zeros(1) for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
            self.y = {(i, j): np.zeros(1) for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
        elif self.mode == 'median':
            # Initialize following reference code exactly - unicast only
            self.z = {(i, j): 0 for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
            self.y = {(i, j): 0 for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}

        self.neighbors = []
        for i in range(self.n):
            neighbors_idx = []
            for j in range(self.n):
                if self.adj_matrix[i, j] == 1:
                    neighbors_idx.append(j)
            self.neighbors.append(neighbors_idx)

    def get_active_nodes(self):
        """Get list of active nodes for current iteration"""
        if self.active_nodes < 1:
            num_active = int(self.active_nodes * self.n)
            return np.random.choice(range(self.n), num_active, replace=False)
        else:
            return range(self.n)

    def calculate_error(self, x_est, x_true):
        """Calculate normalized L2 error"""
        norm_x_true = np.linalg.norm(x_true)
        if norm_x_true == 0:
            return np.linalg.norm(x_est - x_true)
        return np.linalg.norm(x_est - x_true) / norm_x_true

    def step_average(self):
        """Execute one PDMM iteration for average computation"""
        active_nodes = self.get_active_nodes()
        transmission_count = 0

        active_nodes_set = set(active_nodes)

        new_values = self.values.copy()
        for i in active_nodes:
            sum_Az = np.zeros(1)
            for j in self.neighbors[i]:

                A_ij = 1 if i < j else -1
                sum_Az += A_ij * self.z[(i, j)]
            new_values[i] = (self.initial_values[i] - sum_Az[0]) / (1 + self.c * len(self.neighbors[i]))
        self.values = new_values

        y_prev = self.y.copy()
        for i in active_nodes:
            for j in self.neighbors[i]:

                A_ij = 1 if i < j else -1
                self.y[(i, j)] = self.z[(i, j)] + 2 * self.c * A_ij * self.values[i]

        new_z = self.z.copy()
        for i in active_nodes:
            if self.broadcast:
                if self.transmission_loss > 0 and np.random.rand() < self.transmission_loss:
                    transmission_count += 1
                    continue

                # constrain the selected neighbors must be active nodes
                active_neighbors = [j for j in self.neighbors[i] if j in active_nodes_set]
                if active_neighbors:
                    for j in active_neighbors:
                        A_ij = 1 if i < j else -1
                        new_z[(j, i)] = y_prev[(j, i)] + 2 * self.c * A_ij * self.values[i]
                    transmission_count += 1
            else:
                for j in self.neighbors[i]:
                    # constrain the selected neighbors must be active nodes
                    if j not in active_nodes_set:
                        continue

                    if self.transmission_loss > 0 and np.random.rand() < self.transmission_loss:
                        transmission_count += 1
                        continue

                    new_z[(j, i)] = self.y[(i, j)]
                    transmission_count += 1


        self.z = new_z
        self.transmissions += transmission_count
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)

    def step_median(self):
        """Execute one PDMM iteration for median computation - UNICAST ONLY"""
        active_nodes = self.get_active_nodes()
        transmission_count = 0

        # x-update step (following reference code logic exactly)
        for i in active_nodes:
            # Find neighbors
            neighbors_idx = self.neighbors[i]
            if len(neighbors_idx) == 0:
               continue

            # Loop through neighbors
            sum_val = 0
            for j in neighbors_idx:
               A = 1 if i < j else -1
               sum_val += A * self.z.get((i, j), 0)

            # Median-specific x update rule
            val = -sum_val / (self.c * len(neighbors_idx))
            bound = 1 / (self.c * len(neighbors_idx))

            if np.abs(self.initial_values[i] - val) <= bound:
               self.values[i] = self.initial_values[i]
            else:
               self.values[i] = val + np.sign(self.initial_values[i] - val) * bound

            # y update (following reference code structure)
            for j in neighbors_idx:
                A = 1 if i < j else -1
                self.y[(i, j)] = self.z.get((i, j), 0) + 2 * self.c * A * self.values[i]

        # Update auxiliary variables - UNICAST ONLY
        z_prev = self.z.copy()

        for i in active_nodes:
            neighbors_idx = self.neighbors[i]

            # Unicast mode only - simplified z update with theta parameter
            for j in neighbors_idx:
                A = 1 if i < j else -1
                self.z[(j, i)] = (1 - self.theta) * z_prev.get((j, i), 0) + self.theta * self.y.get((i, j), 0)
                transmission_count += 1

        # Update transmission count and history
        self.transmissions += transmission_count
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)


    def step(self):
        """Execute one PDMM iteration based on the selected mode"""
        if self.mode == 'average':
            self.step_average()
        elif self.mode == 'median':
            self.step_median()

    def run(self, max_iterations=1000, tolerance=1e-8):
        """Run PDMM algorithm until convergence"""
        start_time = time.time()
        x_true = self.true_target * np.ones(self.n)  # Define x_true for both modes

        for k in range(max_iterations):
            self.step()
            error = self.calculate_error(self.values, x_true)

            if error < tolerance:
                elapsed_time = time.time() - start_time
                print(f"PDMM ({self.mode}, θ={self.theta}, broadcast={self.broadcast}) converged at iteration {k+1}")
                print(f"Final error: {error:.2e}")
                print(f"Total transmissions: {self.transmissions}")
                print(f"Runtime: {elapsed_time:.2f} seconds")
                break

            if (k + 1) % 100 == 0:
                print(f"PDMM ({self.mode}, θ={self.theta}, broadcast={self.broadcast}) iteration {k+1}: error = {error:.2e}")

        return self.history, self.transmissions, self.transmissions_history

    def compute_error_history(self):
        """Compute normalized L2 error history"""
        history_array = np.array(self.history)
        errors = []
        x_true = self.true_target * np.ones(self.n)  # Define x_true for both modes
        for values in history_array:
            error = self.calculate_error(values, x_true)
            errors.append(error)
        return errors

def plot_all_nodes_e(history, true_average, n_nodes, transmissions_history, max_transmissions=None, title_prefix=""):
    """Plot evolution of all nodes throughout transmissions"""
    history_array = np.array(history)
    trans_array = np.array(transmissions_history)
    n_trans = len(trans_array)
    if max_transmissions is not None:
        idx = np.searchsorted(trans_array, max_transmissions, side='right')
        n_trans = min(n_trans, idx)
        history_array = history_array[:n_trans]
        trans_array = trans_array[:n_trans]

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, (1, 2))
    for i in range(n_nodes):
        ax1.plot(trans_array, history_array[:, i], alpha=0.3, linewidth=0.5)
    ax1.axhline(y=true_average, color='red', linestyle='--',
                linewidth=2, label=f'True average = {true_average:.2f}')
    ax1.set_xlabel('Transmissions')
    ax1.set_ylabel('Node Values')
    ax1.set_title(f'{title_prefix}Evolution of All {n_nodes} Nodes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 3)
    zoom_trans = min(100, n_trans)
    for i in range(n_nodes):
        ax2.plot(trans_array[:zoom_trans], history_array[:zoom_trans, i],
                 alpha=0.3, linewidth=0.5)
    ax2.axhline(y=true_average, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Transmissions')
    ax2.set_ylabel('Node Values')
    ax2.set_title(f'{title_prefix}Zoomed View: First {zoom_trans} Transmissions')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 4)
    errors = np.linalg.norm(history_array - true_average, axis=1) / np.linalg.norm(history_array[0])
    ax3.semilogy(trans_array, errors, 'b-', linewidth=2)
    ax3.set_xlabel('Transmissions')
    ax3.set_ylabel('Normalized L2 Error')
    ax3.set_title(f'{title_prefix}Convergence: Normalized L2 Error vs Transmissions')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_detailed_convergence_analysis(history, true_average, initial_values, transmissions_history, title_prefix=""):
    """Plot detailed convergence analysis"""
    history_array = np.array(history)
    trans_array = np.array(transmissions_history)
    n_trans = len(trans_array)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    errors = np.linalg.norm(history_array - true_average, axis=1) / np.linalg.norm(initial_values)
    ax1.semilogy(trans_array, errors, 'b-', linewidth=2)
    ax1.set_xlabel('Transmissions')
    ax1.set_ylabel('Normalized L2 Error')
    ax1.set_title(f'{title_prefix}Convergence: Normalized L2 Error vs Transmissions')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    min_vals = np.min(history_array, axis=1)
    max_vals = np.max(history_array, axis=1)
    avg_vals = np.mean(history_array, axis=1)
    ax2.plot(trans_array, min_vals, 'b-', label='Min', alpha=0.7)
    ax2.plot(trans_array, max_vals, 'r-', label='Max', alpha=0.7)
    ax2.plot(trans_array, avg_vals, 'g-', label='Average', linewidth=2)
    ax2.axhline(y=true_average, color='black', linestyle='--',
                label='True average', alpha=0.7)
    ax2.set_xlabel('Transmissions')
    ax2.set_ylabel('Values')
    ax2.set_title(f'{title_prefix}Min, Max, and Average Values Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    variance_history = np.var(history_array, axis=1)
    ax3.semilogy(trans_array, variance_history, 'g-', linewidth=2)
    ax3.set_xlabel('Transmissions')
    ax3.set_ylabel('Variance')
    ax3.set_title(f'{title_prefix}Variance Evolution')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    consensus_measure = max_vals - min_vals
    ax4.semilogy(trans_array, consensus_measure, 'm-', linewidth=2)
    ax4.set_xlabel('Transmissions')
    ax4.set_ylabel('Max - Min')
    ax4.set_title(f'{title_prefix}Consensus Measure: Maximum Difference Between Nodes')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def find_optimal_network_parameters(area_size=100, d=2, n_min=155, n_max=200,
                                    max_attempts=50, target_connectivity=0.8):
    """
    Find optimal network parameters:
    - Prints all (n, r) combinations that empirically reach ≥95% connectivity
    - Returns the (n, r) pair that satisfies theoretical 100% connectivity guarantee
    """

    theoretical_result = None

    for n_sensors in range(n_min, n_max + 1):
        # === Theoretical radius based on formula ===
        rhs = 2 * np.log(n_sensors) / n_sensors
        r_norm = rhs ** (1 / d)
        r_theory = r_norm * area_size

        # Save the first theoretical configuration
        if theoretical_result is None:
            theoretical_result = {
                'n_sensors': n_sensors,
                'radius': r_theory
            }

        # === Empirical simulation for 95% connectivity ===
        r_range = np.linspace(max(10, r_theory - 5), r_theory + 10, 10)
        for r in r_range:
            connected_count = 0
            for _ in range(max_attempts):
                network = SensorNetwork(n_sensors=n_sensors,
                                        area_size=area_size,
                                        transmission_radius=r)
                if network.generate_network():
                    connected_count += 1
            connectivity_rate = connected_count / max_attempts
            if connectivity_rate >= target_connectivity:
                print(f"95% connectivity: n={n_sensors}, r={r:.1f}m -> success rate={connectivity_rate:.1%}")
                break

    # === Final output ===
    n_opt = theoretical_result['n_sensors']
    r_opt = theoretical_result['radius']
    print(f"\n Theoretical 100% connectivity configuration: n={n_opt}, r={r_opt:.2f}m")
    return n_opt, r_opt

def analyze_c_vs_transmissions(network, initial_values, c_values, tolerance=1e-13, max_iterations=2000):
    """Analyze PDMM Broadcast performance for different c values"""
    print("\n=== Analyzing c vs Transmissions for PDMM Broadcast ===")
    transmissions = []
    for c in c_values:
        pdmm = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=True, transmission_loss=0, active_nodes=1)
        _, trans_count, _ = pdmm.run(max_iterations=max_iterations, tolerance=tolerance)
        transmissions.append(trans_count)
        print(f"c={c:.2f}: Transmissions={trans_count}")
    return c_values, transmissions

def plot_c_vs_transmissions(c_values, transmissions):
    """Plot c values vs total transmissions for PDMM Broadcast"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c_values, transmissions, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('c Value', fontsize=15)
    ax.set_ylabel('Total Transmissions', fontsize=15)
    ax.set_title('PDMM Broadcast: c Value vs Total Transmissions (Tolerance=1e-13)', fontsize=18)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    optimal_c = c_values[np.argmin(transmissions)]
    optimal_trans = min(transmissions)
    ax.axvline(x=optimal_c, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal c={optimal_c:.2f} (Trans={optimal_trans})')
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig

def analyze_transmission_loss_vs_error(network, initial_values, loss_probs, c=0.1, max_iterations=2000, tolerance=1e-13, broadcast=True):
    """Analyze PDMM performance for different transmission loss probabilities"""
    mode = "Broadcast" if broadcast else "Unicast"
    print(f"\n=== Analyzing Transmission Loss vs Error for PDMM {mode} ===")
    errors_list = []
    trans_list = []
    for p in loss_probs:
        pdmm = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=broadcast,
                    transmission_loss=p, active_nodes=1)
        history, _, trans_history = pdmm.run(max_iterations=max_iterations, tolerance=tolerance)
        errors = pdmm.compute_error_history()
        errors_list.append(errors)
        trans_list.append(trans_history)
        print(f"Transmission Loss={p}: Final Transmissions={trans_history[-1]}, Final Error={errors[-1]:.2e}")
    return loss_probs, errors_list, trans_list

def analyze_active_nodes_vs_error(network, initial_values, active_fractions, c=0.1, max_iterations=2000, tolerance=1e-13, broadcast=False):
    """Analyze PDMM Broadcast performance for different active node fractions"""
    print("\n=== Analyzing Active Nodes vs Error for PDMM Broadcast ===")
    errors_list = []
    trans_list = []
    for a in active_fractions:
        pdmm = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=broadcast,
                    transmission_loss=0, active_nodes=a)
        history, _, trans_history = pdmm.run(max_iterations=max_iterations, tolerance=tolerance)
        errors = pdmm.compute_error_history()
        errors_list.append(errors)
        trans_list.append(trans_history)
        print(f"Active Nodes Fraction={a}: Final Transmissions={trans_history[-1]}, Final Error={errors[-1]:.2e}")
    return active_fractions, errors_list, trans_list

def plot_transmissions_vs_error(params, errors_list, trans_list, param_name, title_prefix):
    """Plot transmissions vs normalized L2 error for different parameter values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'r', 'g', 'm']
    for i, (param, errors, trans) in enumerate(zip(params, errors_list, trans_list)):
        ax.semilogy(trans, errors, f'{colors[i]}-', label=f'{param_name}={param}', linewidth=2)
    ax.set_xlabel('Number of Transmissions', fontsize=15)
    ax.set_ylabel('Normalized L2 Error', fontsize=15)
    ax.set_title(f'{title_prefix}: Transmissions vs Normalized L2 Error (Tolerance=1e-13)', fontsize=18)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig

def plot_transmissions_vs_error_modes(loss_probs, errors_list_unicast, trans_list_unicast, errors_list_broadcast, trans_list_broadcast):
    """Plot transmissions vs normalized L2 error for unicast and broadcast modes in separate subplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['b', 'r', 'g', 'm']

    # Left subplot: Unicast
    for i, (p, errors, trans) in enumerate(zip(loss_probs, errors_list_unicast, trans_list_unicast)):
        ax1.semilogy(trans, errors, f'{colors[i]}-', label=f'Loss={p}', linewidth=2)

    ax1.set_xlabel('Number of Transmissions', fontsize=12)
    ax1.set_ylabel('Normalized L2 Error', fontsize=12)
    ax1.set_title('PDMM Unicast\nTransmission Loss vs Error', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Right subplot: Broadcast
    for i, (p, errors, trans) in enumerate(zip(loss_probs, errors_list_broadcast, trans_list_broadcast)):
        ax2.semilogy(trans, errors, f'{colors[i]}-', label=f'Loss={p}', linewidth=2)

    ax2.set_xlabel('Number of Transmissions', fontsize=12)
    ax2.set_ylabel('Normalized L2 Error', fontsize=12)
    ax2.set_title('PDMM Broadcast\nTransmission Loss vs Error', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=11)

    # Overall figure title
    fig.suptitle('PDMM: Transmission Loss Analysis (Tolerance=1e-13)', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def compare_three_algorithms(network, initial_values, c=0.1, tolerance=1e-8, max_iterations=2000):
    """
    Compare Gossip, PDMM Unicast, and PDMM Broadcast algorithms
    Plot transmission vs normalized L2 error for all three algorithms
    """
    print("\n=== Comparing Gossip, PDMM Unicast, and PDMM Broadcast ===")

    # Run Randomized Gossip
    print("Running Randomized Gossip...")
    gossip = RandomizedGossip(network.adjacency, initial_values)
    gossip.run(max_iterations=max_iterations*100, tolerance=tolerance)  # Gossip may need more iterations
    gossip_errors = gossip.compute_error_history()
    gossip_trans = gossip.transmissions_history

    # Run PDMM Unicast
    print("Running PDMM Unicast...")
    pdmm_unicast = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=False)
    pdmm_unicast.run(max_iterations=max_iterations, tolerance=tolerance)
    unicast_errors = pdmm_unicast.compute_error_history()
    unicast_trans = pdmm_unicast.transmissions_history

    # Run PDMM Broadcast
    print("Running PDMM Broadcast...")
    pdmm_broadcast = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=True)
    pdmm_broadcast.run(max_iterations=max_iterations, tolerance=tolerance)
    broadcast_errors = pdmm_broadcast.compute_error_history()
    broadcast_trans = pdmm_broadcast.transmissions_history

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogy(gossip_trans, gossip_errors, 'b-', linewidth=2, label='Randomized Gossip')
    ax.semilogy(unicast_trans, unicast_errors, 'r-', linewidth=2, label='PDMM Unicast')
    ax.semilogy(broadcast_trans, broadcast_errors, 'g-', linewidth=2, label='PDMM Broadcast')

    ax.set_xlabel('Number of Transmissions', fontsize=14)
    ax.set_ylabel('Normalized L2 Error', fontsize=14)
    ax.set_title(f'Algorithm Comparison: Transmission vs Error (c={c}, tol={tolerance})', fontsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()

    # Print final results
    print(f"Gossip: Final transmissions = {gossip_trans[-1]}, Final error = {gossip_errors[-1]:.2e}")
    print(f"PDMM Unicast: Final transmissions = {unicast_trans[-1]}, Final error = {unicast_errors[-1]:.2e}")
    print(f"PDMM Broadcast: Final transmissions = {broadcast_trans[-1]}, Final error = {broadcast_errors[-1]:.2e}")

    return fig

def compare_median_pdmm_theta(network, initial_values, c=0.1, tolerance=1e-6, max_iterations=2000):
    """
    Compare 1/2avg PDMM vs std PDMM for median consensus - UNICAST ONLY
    Returns transmission vs error curves in two separate subplots
    """
    print(f"\n=== Median PDMM Comparison: 1/2avg vs std (c={c}, tol={tolerance}, UNICAST) ===")

    # Run 1/2avg PDMM (θ=0.5) - UNICAST
    print("Running 1/2avg PDMM (θ=0.5) - Unicast...")
    pdmm_half = PDMM(network.adjacency, initial_values, c=c, mode='median',
                     broadcast=False, theta_mode='1/2-avg')
    pdmm_half.run(max_iterations=50000, tolerance=1e-10)
    errors_half = pdmm_half.compute_error_history()
    trans_half = pdmm_half.transmissions_history

    # Run std PDMM (θ=1.0) - UNICAST
    print("Running std PDMM (θ=1.0) - Unicast...")
    pdmm_std = PDMM(network.adjacency, initial_values, c=c, mode='median',
                    broadcast=False, theta_mode='std')
    pdmm_std.run(max_iterations=max_iterations, tolerance=tolerance)
    errors_std = pdmm_std.compute_error_history()
    trans_std = pdmm_std.transmissions_history

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left subplot: 1/2avg PDMM (θ=0.5)
    ax1.semilogy(trans_half, errors_half, 'b-', linewidth=2)
    ax1.set_xlabel('Number of Transmissions', fontsize=12)
    ax1.set_ylabel('Normalized L2 Error', fontsize=12)
    ax1.set_title('1/2avg PDMM (θ=0.5)\nUnicast Median Consensus', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Right subplot: std PDMM (θ=1.0)
    ax2.semilogy(trans_std, errors_std, 'r-', linewidth=2)
    ax2.set_xlabel('Number of Transmissions', fontsize=12)
    ax2.set_ylabel('Normalized L2 Error', fontsize=12)
    ax2.set_title('std PDMM (θ=1.0)\nUnicast Median Consensus', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Overall figure title
    fig.suptitle(f'Median Consensus Comparison (tol={tolerance})', fontsize=16, y=1.02)
    plt.tight_layout()

    # Print results
    print(f"1/2avg PDMM: Final transmissions = {trans_half[-1]}, Final error = {errors_half[-1]:.2e}")
    print(f"std PDMM: Final transmissions = {trans_std[-1]}, Final error = {errors_std[-1]:.2e}")

    return fig

# Main execution
print("=== Build optimal network parameters ===")
n_sensors, transmission_radius = find_optimal_network_parameters()

network = SensorNetwork(n_sensors=n_sensors, area_size=100, transmission_radius=transmission_radius)
attempts = 0
while not network.generate_network() and attempts < 100:
    attempts += 1

if attempts == 100:
    print("Cannot generate connected network, please adjust parameters")
else:
    print("Connected network generated successfully!")

stats = network.get_network_stats()
print("Network Statistics:")
print(f"  Number of nodes: {stats['nodes']}")
print(f"  Number of edges: {stats['edges']}")
print(f"  Average degree: {stats['average_degree']:.2f}")
print(f"  Network diameter: {stats['diameter']}")
print(f"  Average path length: {stats['average_path_length']:.2f}")
print(f"  Clustering coefficient: {stats['clustering_coefficient']:.3f}")

network_fig = network.visualize_network()
plt.show()

initial_values = np.random.normal(20, 5, n_sensors)
print(f"True average: {np.mean(initial_values):.4f}")
print(f"True median: {np.median(initial_values):.4f}")
print(f"Initial standard deviation: {np.std(initial_values):.4f}")

plt.figure(figsize=(10, 6))
plt.hist(initial_values, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(initial_values), color='red', linestyle='--',
            linewidth=2, label=f'True Average = {np.mean(initial_values):.2f}')
plt.axvline(np.median(initial_values), color='green', linestyle='--',
            linewidth=2, label=f'True Median = {np.median(initial_values):.2f}')
plt.xlabel('Initial Values')
plt.ylabel('Frequency')
plt.title('Distribution of Initial Sensor Measurements')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Perform Randomized Gossip ===")
gossip = RandomizedGossip(network.adjacency, initial_values)
gossip_history, gossip_trans, gossip_trans_history = gossip.run(max_iterations=200000, tolerance=1e-8)

print("\n=== Perform PDMM Unicast ===")
pdmm_unicast = PDMM(network.adjacency, initial_values, c=0.1, mode='average', broadcast=False)
history_unicast, transmissions_unicast, trans_history_unicast = pdmm_unicast.run(max_iterations=2000)

print("\n=== Perform PDMM Broadcast ===")
pdmm_broadcast = PDMM(network.adjacency, initial_values, c=0.1, mode='average', broadcast=True)
history_broadcast, transmissions_broadcast, trans_history_broadcast = pdmm_broadcast.run(max_iterations=2000)

print("\n=== Three Algorithms Comparison ===")
comparison_fig = compare_three_algorithms(network, initial_values, c=0.1, tolerance=1e-8)
plt.show()

print("\n=== Randomized Gossip Analysis ===")
gossip_evolution_fig = plot_all_nodes_e(gossip_history, gossip.true_average, n_sensors, gossip_trans_history, title_prefix="Gossip: ")
plt.show()

gossip_analysis_fig = plot_detailed_convergence_analysis(gossip_history, gossip.true_average, initial_values, gossip_trans_history, title_prefix="Gossip: ")
plt.show()

print("\n=== PDMM Unicast Analysis ===")
pdmm_unicast_evolution_fig = plot_all_nodes_e(history_unicast, pdmm_unicast.true_target, n_sensors, trans_history_unicast, title_prefix="PDMM Unicast: ")
plt.show()

pdmm_unicast_analysis_fig = plot_detailed_convergence_analysis(history_unicast, pdmm_unicast.true_target, initial_values, trans_history_unicast, title_prefix="PDMM Unicast: ")
plt.show()

print("\n=== PDMM Broadcast Analysis ===")
pdmm_broadcast_evolution_fig = plot_all_nodes_e(history_broadcast, pdmm_broadcast.true_target, n_sensors, trans_history_broadcast, title_prefix="PDMM Broadcast: ")
plt.show()

pdmm_broadcast_analysis_fig = plot_detailed_convergence_analysis(history_broadcast, pdmm_broadcast.true_target, initial_values, trans_history_broadcast, title_prefix="PDMM Broadcast: ")
plt.show()

print("\n=== PDMM Broadcast: c vs Transmissions Analysis ===")
c_values = np.linspace(0.01, 2.0, 50)
c_values, transmissions = analyze_c_vs_transmissions(network, initial_values, c_values, tolerance=1e-13)
c_trans_fig = plot_c_vs_transmissions(c_values, transmissions)
plt.show()

print("\n=== Transmission Loss vs Error Analysis (Unicast) ===")
loss_probs = [0.0, 0.25, 0.5, 0.75]
loss_probs, errors_loss_unicast, trans_loss_unicast = analyze_transmission_loss_vs_error(
    network, initial_values, loss_probs, c=0.1, max_iterations=2000, tolerance=1e-13, broadcast=False)

print("\n=== Transmission Loss vs Error Analysis (Broadcast) ===")
loss_probs, errors_loss_broadcast, trans_loss_broadcast = analyze_transmission_loss_vs_error(
    network, initial_values, loss_probs, c=0.1, max_iterations=2000, tolerance=1e-13, broadcast=True)

trans_loss_fig = plot_transmissions_vs_error_modes(
    loss_probs, errors_loss_unicast, trans_loss_unicast, errors_loss_broadcast, trans_loss_broadcast)
plt.show()

print("\n=== Active Nodes vs Error Analysis ===")
active_fractions = [1.0, 0.75, 0.5, 0.25]
active_fractions, errors_active, trans_active = analyze_active_nodes_vs_error(
    network, initial_values, active_fractions, c=0.1, max_iterations=20000, tolerance=1e-12, broadcast=False)
trans_active_fig = plot_transmissions_vs_error(
    active_fractions, errors_active, trans_active, "Active Nodes Fraction", "PDMM Unicast")
plt.show()

print("\n=== Median PDMM Theta Comparison (Unicast Only) ===")
median_comparison_fig = compare_median_pdmm_theta(network, initial_values, c=0.13, tolerance=1e-6)
plt.show()