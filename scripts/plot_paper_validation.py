#!/usr/bin/env python3
"""Plotting script for paper validation and end-to-end BKZ benchmarks.

Generates:
1. data/figures/fig_paper_validation.pdf (.png) - 3-panel plot replicating Figure 5 of Dachman-Soled et al. (CRYPTO 2020)
2. data/figures/fig_e2e_bkz_recovery.pdf (.png) - 2-panel plot showing wall-clock BKZ reduction time and tours vs hint count.
"""

import csv
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "font.family": "serif",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
})

def plot_paper_validation(decay_csv, out_pdf, out_png):
    data = {"Perfect": ([], [], []), "Modular": ([], [], []), "Approximate": ([], [], [])}
    
    with open(decay_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ht = row["hint_type"]
            if ht == "Baseline":
                for k in data:
                    data[k][0].append(0)
                    data[k][1].append(float(row["beta_gsa"]))
                    data[k][2].append(float(row["beta_prob"]))
            elif ht in data:
                data[ht][0].append(int(row["count"]))
                data[ht][1].append(float(row["beta_gsa"]))
                data[ht][2].append(float(row["beta_prob"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)

    configs = [
        ("Perfect", axes[0], "# Perfect Hints (revealing coordinates)", "Perfect Hints\n(Exact secret reveals)"),
        ("Modular", axes[1], "# Modular Hints (mod 11)", "Modular Hints\n(mod 11, random weight 5)"),
        ("Approximate", axes[2], "# Approximate Hints (sigma_e^2 = 3)", "Approximate Hints\n(sigma_e^2 = 3, weight 5)"),
    ]

    for ht, ax, xlabel, subtitle in configs:
        counts, gsa, prob = data[ht]
        # Sort by count
        pts = sorted(zip(counts, gsa, prob), key=lambda x: x[0])
        x = [p[0] for p in pts]
        y_gsa = [p[1] for p in pts]
        y_prob = [p[2] for p in pts]

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.plot(x, y_gsa, marker="o", color="#2a78d6", linewidth=1.8, markersize=5, label="GSA-Intersect Prediction")
        ax.plot(x, y_prob, marker="^", color="#eb6834", linewidth=1.8, markersize=5, linestyle="--", label="Probabilistic Simulator")

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold", pad=10, color=INK)
        ax.set_xlabel(xlabel, fontsize=9.5, labelpad=8, color=INK)
        ax.set_ylim(-1, 50)
        ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=AXIS, fontsize=8.5)

    axes[0].set_ylabel("Required BKZ Block Size beta", fontsize=10, labelpad=8, color=INK)
    plt.suptitle("Replication of Dachman-Soled et al. (CRYPTO 2020) Figure 5: Security Decay Under Hints\n(Instance: n = m = 70, q = 3301, sigma^2 = 20, Kannan dim = 141, baseline beta = 46)",
                 fontsize=11.5, fontweight="bold", y=1.03, color=INK)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_pdf} and {out_png}")

def plot_e2e_bkz_recovery(e2e_csv, out_pdf, out_png):
    hints = []
    times_ms = []
    tours = []
    succ_rates = []

    with open(e2e_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hints.append(int(row["k"]))
            times_ms.append(float(row["mean_time_ms"]))
            tours.append(float(row["mean_tours"]))
            succ_rates.append(float(row["success_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel 1: Wall-Clock Runtime
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.plot(hints, times_ms, marker="o", color="#2a78d6", linewidth=2.0, markersize=6, label="Mean BKZ-4 Wall Time")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Perfect Coordinate Hints (k)", fontsize=10, labelpad=8, color=INK)
    ax1.set_ylabel("Wall-Clock Execution Time (ms) [Log Scale]", fontsize=10, labelpad=8, color=INK)
    ax1.set_title("Reduction Wall-Clock Runtime Speedup\n(Kannan dim = 25 -> 13)", fontsize=11, fontweight="bold", pad=10, color=INK)

    if len(times_ms) > 0:
        ax1.annotate(f"{times_ms[0]:.1f} ms\n(0 hints)", (hints[0], times_ms[0]),
                     textcoords="offset points", xytext=(10, 5), fontsize=8.5, color="#2a78d6", fontweight="bold")
        ax1.annotate(f"{times_ms[-1]:.1f} ms\n({times_ms[0]/times_ms[-1]:.0f}x faster)", (hints[-1], times_ms[-1]),
                     textcoords="offset points", xytext=(-50, 10), fontsize=8.5, color="#2a78d6", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=AXIS, fontsize=9)

    # Panel 2: BKZ Tours
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.plot(hints, tours, marker="s", color="#eb6834", linewidth=2.0, markersize=6, label="Mean Tours to Convergence")
    ax2.set_xlabel("Number of Perfect Coordinate Hints (k)", fontsize=10, labelpad=8, color=INK)
    ax2.set_ylabel("BKZ-4 Tours Completed", fontsize=10, labelpad=8, color=INK)
    ax2.set_title("Lattice Reduction Convergence Tours\n(100% Secret Recovery across all trials)", fontsize=11, fontweight="bold", pad=10, color=INK)
    ax2.set_ylim(0, 18)
    ax2.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=AXIS, fontsize=9)

    plt.suptitle("End-to-End Secret Recovery under BKZ Reduction\n(Tractable Instance: n = m = 12, q = 257, CBD(3), 15 trials per point)",
                 fontsize=11.5, fontweight="bold", y=1.03, color=INK)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_pdf} and {out_png}")

if __name__ == "__main__":
    os.makedirs("data/figures", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    decay_csv = "data/paper_validation_decay.csv"
    e2e_csv = "data/e2e_bkz_recovery.csv"

    fig_val_pdf = "data/figures/fig_paper_validation.pdf"
    fig_val_png = "data/figures/fig_paper_validation.png"
    fig_e2e_pdf = "data/figures/fig_e2e_bkz_recovery.pdf"
    fig_e2e_png = "data/figures/fig_e2e_bkz_recovery.png"

    plot_paper_validation(decay_csv, fig_val_pdf, fig_val_png)
    plot_e2e_bkz_recovery(e2e_csv, fig_e2e_pdf, fig_e2e_png)

    # Copy to report/figures
    shutil.copy(fig_val_pdf, "report/figures/fig_paper_validation.pdf")
    shutil.copy(fig_val_png, "report/figures/fig_paper_validation.png")
    shutil.copy(fig_e2e_pdf, "report/figures/fig_e2e_bkz_recovery.pdf")
    shutil.copy(fig_e2e_png, "report/figures/fig_e2e_bkz_recovery.png")
    print("[OK] Copied figures to report/figures/")

