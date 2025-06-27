import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import networkx as nx
from matplotlib.patches import Circle

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