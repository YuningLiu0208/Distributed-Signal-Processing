import numpy as np
from sensor_network import SensorNetwork
from pdmm import PDMM
from randomized_gossip import RandomizedGossip
import matplotlib.pyplot as plt

def find_optimal_network_parameters(area_size=100, d=2, n_min=155, n_max=200,
                                    max_attempts=50, target_connectivity=0.8):
    """
    Find optimal network parameters:
    - Prints all (n, r) combinations that empirically reach ≥95% connectivity
    - Returns the (n, r) pair that satisfies theoretical 100% connectivity guarantee
    """
    theoretical_result = None
    for n_sensors in range(n_min, n_max + 1):
        rhs = 2 * np.log(n_sensors) / n_sensors
        r_norm = rhs ** (1 / d)
        r_theory = r_norm * area_size
        if theoretical_result is None:
            theoretical_result = {
                'n_sensors': n_sensors,
                'radius': r_theory
            }
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

def compare_three_algorithms(network, initial_values, c=0.1, tolerance=1e-8, max_iterations=2000):
    """
    Compare Gossip, PDMM Unicast, and PDMM Broadcast algorithms
    Plot transmission vs normalized L2 error for all three algorithms
    """
    print("\n=== Comparing Gossip, PDMM Unicast, and PDMM Broadcast ===")
    print("Running Randomized Gossip...")
    gossip = RandomizedGossip(network.adjacency, initial_values)
    gossip.run(max_iterations=max_iterations*100, tolerance=tolerance)
    gossip_errors = gossip.compute_error_history()
    gossip_trans = gossip.transmissions_history
    print("Running PDMM Unicast...")
    pdmm_unicast = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=False)
    pdmm_unicast.run(max_iterations=max_iterations, tolerance=tolerance)
    unicast_errors = pdmm_unicast.compute_error_history()
    unicast_trans = pdmm_unicast.transmissions_history
    print("Running PDMM Broadcast...")
    pdmm_broadcast = PDMM(network.adjacency, initial_values, c=c, mode='average', broadcast=True)
    pdmm_broadcast.run(max_iterations=max_iterations, tolerance=tolerance)
    broadcast_errors = pdmm_broadcast.compute_error_history()
    broadcast_trans = pdmm_broadcast.transmissions_history
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
    print("Running 1/2avg PDMM (θ=0.5) - Unicast...")
    pdmm_half = PDMM(network.adjacency, initial_values, c=c, mode='median',
                     broadcast=False, theta_mode='1/2-avg')
    pdmm_half.run(max_iterations=50000, tolerance=1e-10)
    errors_half = pdmm_half.compute_error_history()
    trans_half = pdmm_half.transmissions_history
    print("Running std PDMM (θ=1.0) - Unicast...")
    pdmm_std = PDMM(network.adjacency, initial_values, c=c, mode='median',
                    broadcast=False, theta_mode='std')
    pdmm_std.run(max_iterations=max_iterations, tolerance=tolerance)
    errors_std = pdmm_std.compute_error_history()
    trans_std = pdmm_std.transmissions_history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.semilogy(trans_half, errors_half, 'b-', linewidth=2)
    ax1.set_xlabel('Number of Transmissions', fontsize=12)
    ax1.set_ylabel('Normalized L2 Error', fontsize=12)
    ax1.set_title('1/2avg PDMM (θ=0.5)\nUnicast Median Consensus', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(trans_std, errors_std, 'r-', linewidth=2)
    ax2.set_xlabel('Number of Transmissions', fontsize=12)
    ax2.set_ylabel('Normalized L2 Error', fontsize=12)
    ax2.set_title('std PDMM (θ=1.0)\nUnicast Median Consensus', fontsize=14)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f'Median Consensus Comparison (tol={tolerance})', fontsize=16, y=1.02)
    plt.tight_layout()
    print(f"1/2avg PDMM: Final transmissions = {trans_half[-1]}, Final error = {errors_half[-1]:.2e}")
    print(f"std PDMM: Final transmissions = {trans_std[-1]}, Final error = {errors_std[-1]:.2e}")
    return fig