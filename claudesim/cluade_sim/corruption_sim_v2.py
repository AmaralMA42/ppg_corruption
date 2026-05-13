"""
Corruption Game — Spatial PGG with Corrupt (P) strategy and Bribery
====================================================================
Optimized: full payoff recomputed once per MCS (vectorised numpy).
Much faster than per-flip local updates without numba.

Model from the paper (main.tex):
    Strategies: C=0, D=1, P=2
    Group size G=5 (site + 4 von Neumann neighbours)
    Each site participates in 5 overlapping groups.

    Public good per group:
        L = r·c·(Nc + Np - σ·Np) / G

    Payoffs with bribery (summed over all 5 groups the focal site belongs to):
        π_C = Σ [ L - c + α·σ·c·(Np/Nc) ]  (bribe term only if Nc>0)
        π_D = Σ [ L ]
        π_P = Σ [ L - c + (1-α)·σ·c ]  if Nc>0
            = Σ [ L - c + σ·c ]         if Nc=0

    Fermi update:  P(i→j) = 1 / (1 + exp(-(π_j - π_i)/k))
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

OUT = Path("/mnt/user-data/outputs")
OUT.mkdir(exist_ok=True)

COLORS = {"C": "#2166ac", "D": "#d73027", "P": "#4dac26"}

# ─────────────────────────────────────────────
# LATTICE
# ─────────────────────────────────────────────

def build_neighbours(L):
    N = L * L
    idx = np.arange(N)
    row, col = idx // L, idx % L
    up    = ((row - 1) % L) * L + col
    down  = ((row + 1) % L) * L + col
    right = row * L + (col + 1) % L
    left  = row * L + (col - 1) % L
    return np.stack([up, right, down, left], axis=1).astype(np.int32)


# ─────────────────────────────────────────────
# FULL VECTORISED PAYOFF
# ─────────────────────────────────────────────

def compute_all_payoffs(strat, viz, r, c, sigma, alpha, G=5):
    """
    Full vectorised payoff for all N sites.
    Each site i is focal in 5 groups: the one centred on i,
    and the ones centred on each of its 4 neighbours.
    """
    N = len(strat)
    # group members for group centred at site g:  g + its 4 neighbours
    # We precompute for every possible group centre the Nc, Np of that group.

    # For each site g, compute Nc_g and Np_g (composition of group centred at g)
    is_C = (strat == 0).astype(np.float32)
    is_P = (strat == 2).astype(np.float32)

    # Sum over the 5 members of the group centred at g
    Nc_grp = is_C + is_C[viz[:, 0]] + is_C[viz[:, 1]] + is_C[viz[:, 2]] + is_C[viz[:, 3]]
    Np_grp = is_P + is_P[viz[:, 0]] + is_P[viz[:, 1]] + is_P[viz[:, 2]] + is_P[viz[:, 3]]

    L_grp = r * c * (Nc_grp + Np_grp - sigma * Np_grp) / G   # shape (N,)

    # For each focal site i, it participates in 5 groups:
    # centred at i, viz[i,0], viz[i,1], viz[i,2], viz[i,3]
    # group indices for focal site i:
    g0 = np.arange(N)
    g1, g2, g3, g4 = viz[:, 0], viz[:, 1], viz[:, 2], viz[:, 3]

    # Sum of L over the 5 groups (everyone gets this base)
    sum_L = L_grp[g0] + L_grp[g1] + L_grp[g2] + L_grp[g3] + L_grp[g4]

    payoff = sum_L.copy()

    # ── Cooperators (C = 0) ──────────────────────────────────────────────
    mask_C = (strat == 0)
    # Each group contributes: -c + bribe (if Nc_grp > 0)
    # bribe from group g = α·σ·c·Np_grp[g] / Nc_grp[g]
    for g in [g0, g1, g2, g3, g4]:
        nc_g = Nc_grp[g]
        np_g = Np_grp[g]
        safe_nc = np.where(nc_g > 0, nc_g, 1.0)
        bribe = np.where(nc_g > 0, alpha * sigma * c * np_g / safe_nc, 0.0)
        payoff -= mask_C * c          # cost
        payoff += mask_C * bribe      # bribe
    # We subtracted c 5 times (once per group) — correct: each group they pay c

    # ── Corrupt players (P = 2) ───────────────────────────────────────────
    mask_P = (strat == 2)
    for g in [g0, g1, g2, g3, g4]:
        nc_g = Nc_grp[g]
        steal = np.where(nc_g > 0, (1 - alpha) * sigma * c, sigma * c)
        payoff -= mask_P * c          # cost
        payoff += mask_P * steal      # stolen amount

    return payoff


# ─────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────

def run_single(L, r, c, sigma, alpha, k, G, total_steps, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = L * L
    viz = build_neighbours(L)
    strat = np.random.randint(0, 3, size=N).astype(np.int32)

    ts = np.zeros((3, total_steps))

    for t in range(total_steps):
        payoff = compute_all_payoffs(strat, viz, r, c, sigma, alpha, G)

        counts = np.bincount(strat, minlength=3)
        ts[:, t] = counts / N

        # Fermi update — vectorised
        focal  = np.random.randint(0, N, size=N)
        nb_dir = np.random.randint(0, 4, size=N)
        neigh  = viz[focal, nb_dir]
        rands  = np.random.random(N)

        diff = payoff[neigh] - payoff[focal]
        # numerically stable Fermi
        prob = np.where(diff >= 0,
                        1.0 / (1.0 + np.exp(-diff / k)),
                        np.exp(diff / k) / (1.0 + np.exp(diff / k)))

        should_flip = (strat[focal] != strat[neigh]) & (rands < prob)
        strat[focal[should_flip]] = strat[neigh[should_flip]]

    return ts, strat.reshape(L, L)


def run_samples(L, r, c, sigma, alpha, k, G, total_steps, n_samples, eq_frac=0.5, base_seed=0):
    all_ts = []
    for s in range(n_samples):
        ts, _ = run_single(L, r, c, sigma, alpha, k, G, total_steps, seed=base_seed + s*17)
        all_ts.append(ts)
    all_ts = np.array(all_ts)   # (n_samp, 3, steps)
    eq = int(eq_frac * total_steps)
    steady = all_ts[:, :, eq:].mean(axis=2)
    mean = steady.mean(axis=0)
    sem  = steady.std(axis=0, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else np.zeros(3)
    return mean, sem, all_ts


# ─────────────────────────────────────────────
# PHASE SWEEP
# ─────────────────────────────────────────────

def phase_sweep(alpha_list, r_vals, sigma_vals,
                L=50, c=1.0, k=0.1, G=5,
                total_steps=400, n_samples=3, eq_frac=0.55,
                base_seed=42):
    nr, ns, na = len(r_vals), len(sigma_vals), len(alpha_list)
    results = np.zeros((na, 3, nr, ns))
    total_pts = na * nr * ns
    done = 0
    t0 = time.time()

    for ai, alpha in enumerate(alpha_list):
        for ri, r in enumerate(r_vals):
            for si, sig in enumerate(sigma_vals):
                mean, _, _ = run_samples(L, r, c, sig, alpha, k, G,
                                         total_steps, n_samples, eq_frac, base_seed)
                results[ai, :, ri, si] = mean
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total_pts - done) if done else 0
                if done % 10 == 0 or done <= 5:
                    print(f"  [{done:4d}/{total_pts}  ETA {eta:5.0f}s]  "
                          f"α={alpha:.2f} r={r:.2f} σ={sig:.2f}  "
                          f"C={mean[0]:.2f} D={mean[1]:.2f} P={mean[2]:.2f}")
    return results


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_phase_diagrams(results, alpha_list, r_vals, sigma_vals):
    na = len(alpha_list)
    fig, axes = plt.subplots(na, 3, figsize=(14, 4.2 * na))
    if na == 1:
        axes = axes[np.newaxis, :]

    cmaps    = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
    strat_lbl = ["C (cooperator)", "D (defector)", "P (corrupt)"]

    for ai, alpha in enumerate(alpha_list):
        for si in range(3):
            ax = axes[ai, si]
            data = results[ai, si, :, :]
            im = ax.imshow(data, origin='lower', aspect='auto',
                           vmin=0, vmax=1, cmap=cmaps[si],
                           extent=[sigma_vals[0], sigma_vals[-1],
                                   r_vals[0],     r_vals[-1]])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_title(f"{strat_lbl[si]},  α={alpha:.2f}", fontsize=10, fontweight='bold')
            ax.set_xlabel("σ (corruption strength)", fontsize=9)
            ax.set_ylabel("r (synergy factor)", fontsize=9)
            # mark the σ=1 boundary
            ax.axvline(x=1.0, color='white', lw=1.0, ls='--', alpha=0.7)

    fig.suptitle("Corruption Game — Steady-state strategy fractions\n"
                 "Phase diagram in (σ, r) space", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_r_sweeps(alpha_list, r_vals, sigma_list, results_all):
    """
    results_all: dict alpha → (3, nr, ns_pts) array
    sigma_list: the sigma values used as columns in results_all
    Plot one panel per (alpha, sigma).
    """
    n_alpha = len(alpha_list)
    n_sig   = len(sigma_list)
    fig, axes = plt.subplots(n_alpha, n_sig,
                             figsize=(5 * n_sig, 4 * n_alpha), squeeze=False,
                             sharey=True)

    for ai, alpha in enumerate(alpha_list):
        for sj, (sig, res) in enumerate(zip(sigma_list, results_all[alpha])):
            ax = axes[ai, sj]
            ax.plot(r_vals, res[0], color=COLORS["C"], lw=2, label="C")
            ax.plot(r_vals, res[1], color=COLORS["D"], lw=2, label="D")
            ax.plot(r_vals, res[2], color=COLORS["P"], lw=2, label="P")
            ax.set_xlabel("r", fontsize=11)
            ax.set_title(f"α={alpha:.2f},  σ={sig:.2f}", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if sj == 0:
                ax.set_ylabel("Steady-state fraction", fontsize=10)
            if ai == 0 and sj == 0:
                ax.legend(fontsize=9)

    fig.suptitle("Strategy fractions vs synergy r\n(sweeps for different σ and α)",
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_timeseries_grid(runs, labels):
    n = len(runs)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (all_ts, lbl, r, sig, alpha) in enumerate(runs):
        ax = axes[idx]
        mean_ts = all_ts.mean(axis=0)
        for ts in all_ts:
            ax.plot(ts[0], color=COLORS["C"], alpha=0.15, lw=0.8)
            ax.plot(ts[1], color=COLORS["D"], alpha=0.15, lw=0.8)
            ax.plot(ts[2], color=COLORS["P"], alpha=0.15, lw=0.8)
        ax.plot(mean_ts[0], color=COLORS["C"], lw=2.2, label="C")
        ax.plot(mean_ts[1], color=COLORS["D"], lw=2.2, label="D")
        ax.plot(mean_ts[2], color=COLORS["P"], lw=2.2, label="P")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{lbl}\nr={r}, σ={sig}, α={alpha}", fontsize=9, fontweight='bold')
        ax.set_xlabel("MCS")
        ax.set_ylabel("ρ")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Time evolution at selected parameter points", fontsize=13)
    plt.tight_layout()
    return fig


def plot_snapshots(snap_runs):
    n = len(snap_runs)
    cmap = plt.matplotlib.colors.ListedColormap([COLORS["C"], COLORS["D"], COLORS["P"]])
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))

    for idx, (grid, lbl, r, sig, alpha) in enumerate(snap_runs):
        ax = axes[idx] if n > 1 else axes
        im = ax.imshow(grid, vmin=0, vmax=2, cmap=cmap, interpolation='none')
        ax.set_title(f"{lbl}\nr={r}, σ={sig}, α={alpha}", fontsize=9, fontweight='bold')
        ax.axis('off')

    cbar = plt.colorbar(im, ax=axes, ticks=[0, 1, 2], fraction=0.015, pad=0.02)
    cbar.ax.set_yticklabels(["C", "D", "P"], fontsize=11)
    fig.suptitle("Spatial configuration (final state)", fontsize=12)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    c, k, G = 1.0, 0.1, 5
    L_phase  = 50
    L_ts     = 60
    L_snap   = 80

    # ── 1. Phase diagram ─────────────────────────────────────────────────
    r_vals     = np.linspace(2.5, 6.0, 15)
    sigma_vals = np.linspace(0.5, 3.0, 15)
    alpha_list = [0.0, 0.2, 0.5]

    print("=" * 65)
    print("PHASE DIAGRAM  (σ × r)  for α =", alpha_list)
    print(f"  Grid {len(r_vals)}×{len(sigma_vals)}, L={L_phase}, 3 samples, 400 MCS")
    print("=" * 65)
    t0 = time.time()

    results = phase_sweep(
        alpha_list, r_vals, sigma_vals,
        L=L_phase, c=c, k=k, G=G,
        total_steps=400, n_samples=3, eq_frac=0.55
    )
    print(f"Phase sweep done in {time.time()-t0:.1f}s")

    fig1 = plot_phase_diagrams(results, alpha_list, r_vals, sigma_vals)
    fig1.savefig(OUT / "phase_diagram.png", dpi=150, bbox_inches='tight')
    print("→ Saved phase_diagram.png")

    # ── 2. r-sweep slices ────────────────────────────────────────────────
    # Extract slices from phase diagram at specific sigma values
    sigma_slices = [1.05, 1.5, 2.0]
    results_r = {}   # alpha → list of (3, nr) arrays, one per sigma slice
    for ai, alpha in enumerate(alpha_list):
        results_r[alpha] = []
        for sig in sigma_slices:
            si = np.argmin(np.abs(sigma_vals - sig))
            results_r[alpha].append(results[ai, :, :, si])   # (3, nr)

    fig2 = plot_r_sweeps(alpha_list, r_vals, sigma_slices, results_r)
    fig2.savefig(OUT / "r_sweep.png", dpi=150, bbox_inches='tight')
    print("→ Saved r_sweep.png")

    # ── 3. Time series — interesting points ──────────────────────────────
    print("\nTime series at selected points ...")
    interesting = [
        # (r, sigma, alpha, label)
        (3.5,  1.05, 0.20, "Cyclic / oscillations"),
        (3.5,  2.0,  0.0,  "P dominates (no bribery)"),
        (3.5,  2.0,  0.30, "Coexistence + bribery"),
        (4.5,  1.5,  0.50, "High bribery — C favoured"),
        (5.5,  1.5,  0.20, "High r — C vs P coexistence"),
        (2.8,  1.2,  0.0,  "Low r, P barely beats D"),
    ]

    ts_runs = []
    snap_runs = []
    for r, sig, alpha, lbl in interesting:
        print(f"  {lbl}: r={r}, σ={sig}, α={alpha}")
        mean, sem, all_ts = run_samples(L_ts, r, c, sig, alpha, k, G,
                                        total_steps=800, n_samples=6,
                                        eq_frac=0.5, base_seed=1234)
        ts_runs.append((all_ts, lbl, r, sig, alpha))
        _, grid = run_single(L_snap, r, c, sig, alpha, k, G,
                             total_steps=800, seed=42)
        snap_runs.append((grid, lbl, r, sig, alpha))

    fig3 = plot_timeseries_grid(ts_runs, [x[3] for x in interesting])
    fig3.savefig(OUT / "timeseries.png", dpi=150, bbox_inches='tight')
    print("→ Saved timeseries.png")

    fig4 = plot_snapshots(snap_runs)
    fig4.savefig(OUT / "snapshots.png", dpi=150, bbox_inches='tight')
    print("→ Saved snapshots.png")

    print("\n✓  All figures saved.")
