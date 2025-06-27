import numpy as np
import matplotlib.pyplot as plt

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
    for i, (p, errors, trans) in enumerate(zip(loss_probs, errors_list_unicast, trans_list_unicast)):
        ax1.semilogy(trans, errors, f'{colors[i]}-', label=f'Loss={p}', linewidth=2)
    ax1.set_xlabel('Number of Transmissions', fontsize=12)
    ax1.set_ylabel('Normalized L2 Error', fontsize=12)
    ax1.set_title('PDMM Unicast\nTransmission Loss vs Error', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=11)
    for i, (p, errors, trans) in enumerate(zip(loss_probs, errors_list_broadcast, trans_list_broadcast)):
        ax2.semilogy(trans, errors, f'{colors[i]}-', label=f'Loss={p}', linewidth=2)
    ax2.set_xlabel('Number of Transmissions', fontsize=12)
    ax2.set_ylabel('Normalized L2 Error', fontsize=12)
    ax2.set_title('PDMM Broadcast\nTransmission Loss vs Error', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=11)
    fig.suptitle('PDMM: Transmission Loss Analysis (Tolerance=1e-13)', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig