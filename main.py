import numpy as np
import matplotlib.pyplot as plt
from sensor_network import SensorNetwork
from randomized_gossip import RandomizedGossip
from pdmm import PDMM
from plotting_utils import (plot_all_nodes_e, plot_detailed_convergence_analysis,
                           plot_c_vs_transmissions, plot_transmissions_vs_error,
                           plot_transmissions_vs_error_modes)
from analysis_utils import (find_optimal_network_parameters, analyze_c_vs_transmissions,
                           analyze_transmission_loss_vs_error, analyze_active_nodes_vs_error,
                           compare_three_algorithms, compare_median_pdmm_theta)

np.random.seed(42)

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