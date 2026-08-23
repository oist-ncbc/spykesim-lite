import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle


def plot_simmat(simmat, export_fig_dir, suffix):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax[0].imshow(simmat, cmap='viridis', aspect='auto')
    ax[0].set_title('Edit similarity (with diagonal)')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Index')
    plt.colorbar(im1, ax=ax[0], label='Similarity')

    simmat_no_diag = simmat.copy()
    np.fill_diagonal(simmat_no_diag, np.nan)

    im2 = ax[1].imshow(simmat_no_diag, cmap='viridis', aspect='auto')
    ax[1].set_title('Edit similarity (without diagonal)')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Index')
    plt.colorbar(im2, ax=ax[1], label='Similarity')

    plt.tight_layout()
    fig.savefig(os.path.join(export_fig_dir, f"simmat_{suffix}.png"), dpi=300)
    plt.close(fig)
    

def plot_eligmat(eligmat, norm_simmat, export_fig_dir, suffix):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax[0].imshow(eligmat, cmap='viridis', aspect='auto')
    ax[0].set_title('Eligibility matrix')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Index')
    plt.colorbar(im1, ax=ax[0], label='Eligibility')

    mask = eligmat & np.isfinite(norm_simmat)
    floor = np.percentile(norm_simmat[mask], 1)
    simmat_masked = np.where(eligmat, norm_simmat, floor)

    simmat_masked_no_diag = simmat_masked.copy()
    np.fill_diagonal(simmat_masked_no_diag, np.nan)

    im2 = ax[1].imshow(simmat_masked_no_diag, cmap='viridis', aspect='auto')
    ax[1].set_title('Edit similarity (normalized, masked, without diagonal)')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Index')
    plt.colorbar(im2, ax=ax[1], label='Similarity')

    plt.tight_layout()
    fig.savefig(os.path.join(export_fig_dir, f"eligmat_{suffix}.png"), dpi=300)
    plt.close(fig)


def plot_simmat_vs_norm(simmat, norm_simmat, export_fig_dir, suffix):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    simmat_no_diag = simmat.copy()
    np.fill_diagonal(simmat_no_diag, np.nan)

    im1 = ax[0].imshow(simmat_no_diag, cmap='viridis', aspect='auto')
    ax[0].set_title('Edit similarity (without diagonal)')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Index')
    plt.colorbar(im1, ax=ax[0], label='Similarity')

    normsimmat_no_diag = norm_simmat.copy()
    np.fill_diagonal(normsimmat_no_diag, np.nan)

    im2 = ax[1].imshow(normsimmat_no_diag, cmap='viridis', aspect='auto')
    ax[1].set_title('Edit similarity (normalized, without diagonal)')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Index')
    plt.colorbar(im2, ax=ax[1], label='Similarity')
    
    plt.tight_layout()
    fig.savefig(os.path.join(export_fig_dir, f"norm_simmat_{suffix}.png"), dpi=300)
    plt.close(fig)


def plot_firing_rate(windows, window_ms, neuron_list, times_ms, end_elect_1, export_fig_dir, suffix):
    n_electrode = 10
    offset = 20.0
    pop_thr_Hz = 3.0
    sigma_windows = 2.0

    n_neurons = windows[0].shape[0]
    T = len(windows)

    firing_rate = np.zeros((T, n_neurons), dtype=float)
    firing_rate_pop = np.zeros(T, dtype=float)

    for i, window in enumerate(windows):
        win = window.toarray()
        firing_rate[i] = win.sum(axis=1) * 1000.0 / window_ms
        firing_rate_pop[i] = win.sum() / n_neurons * 1000.0 / window_ms

    firing_rate = firing_rate.T
    firing_rate_smooth = gaussian_filter1d(firing_rate, sigma=sigma_windows, axis=1, mode="nearest")
    firing_rate_pop_smooth = gaussian_filter1d(firing_rate_pop, sigma=sigma_windows, mode="nearest")

    # indices for electrode 1 (first 5)
    idx_e1 = np.arange(min(n_electrode, n_neurons), dtype=int)

    # indices for electrode 2 (first 5 after end_elect_1)
    idx_e2_all = np.where(np.asarray(neuron_list) > end_elect_1)[0]
    idx_e2 = idx_e2_all[:n_electrode]

    plot_idx = np.concatenate([idx_e1, idx_e2])
    colors = (["tab:cyan"] * len(idx_e1)) + (["tab:pink"] * len(idx_e2))

    n_plot = len(plot_idx)

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_plot))

    ymin = -5.0
    ymax = offset * n_plot

    # population activity shading
    dx = np.median(np.diff(times_ms)) if T > 1 else window_ms
    for i in range(T):
        if firing_rate_pop_smooth[i] > pop_thr_Hz:
            ax.add_patch(Rectangle((times_ms[i], ymin), dx, ymax - ymin, alpha=0.3, color="gray"))

    # plot traces: electrode 1 on top, electrode 2 below
    for k, (ridx, col) in enumerate(zip(plot_idx, colors)):
        y = firing_rate_smooth[ridx] + (n_plot - 1 - k) * offset
        ax.plot(times_ms, y, color=col, linewidth=1.2)
        ax.text(times_ms[0], (n_plot - 1 - k) * offset, str(neuron_list[ridx]), va="center", ha="right", fontsize=10)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time (ms)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    fig.savefig(os.path.join(export_fig_dir, f"firing_rate_{suffix}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)