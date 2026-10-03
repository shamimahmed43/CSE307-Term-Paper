"""
plots.py - Publication-Quality Figure Generation

Generates clean, aesthetic figures for the term paper:
1. page_faults_before_after.png: Grouped bar chart comparing page faults.
2. hit_ratio_before_after.png: Grouped bar chart comparing hit ratios.
3. overall_comparison.png: Overall comparison across algorithms.
4. workload_distribution.png: Visual inspection of the workload shift.
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# Academic / Modern Color Palette
PALETTE = {
    "FIFO": "#4A90E2",     # Muted Blue
    "LRU": "#50E3C2",      # Teal / Cyan
    "Learned": "#F5A623",  # Amber / Warm Orange
    "Optimal": "#9013FE"   # Purple / Violet
}


def plot_page_faults_before_after(summary_df: pd.DataFrame, output_path: str) -> None:
    """Generate grouped bar chart comparing page fault counts before and after the shift."""
    df_before = summary_df[summary_df["segment"] == "before_shift"].set_index("algorithm")
    df_after = summary_df[summary_df["segment"] == "after_shift"].set_index("algorithm")

    algorithms = ["FIFO", "LRU", "Learned", "Optimal"]
    # Reindex to ensure consistent ordering
    df_before = df_before.reindex(algorithms)
    df_after = df_after.reindex(algorithms)

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    rects1 = ax.bar(x - width/2, df_before["page_faults"], width, label="Phase 1 (Locality-Heavy)", 
                    color="#4A90E2", edgecolor="#2C3E50", alpha=0.9, linewidth=1)
    rects2 = ax.bar(x + width/2, df_after["page_faults"], width, label="Phase 2 (Random / Bursty)", 
                    color="#E74C3C", edgecolor="#2C3E50", alpha=0.9, linewidth=1)

    ax.set_ylabel("Number of Page Faults", fontsize=12, fontweight="bold")
    ax.set_title("Page Faults Before vs. After Workload Shift", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8F9F9", edgecolor="#BDC3C7", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Attach value labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(height)}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_hit_ratio_before_after(summary_df: pd.DataFrame, output_path: str) -> None:
    """Generate grouped bar chart comparing hit ratio before and after the shift."""
    df_before = summary_df[summary_df["segment"] == "before_shift"].set_index("algorithm")
    df_after = summary_df[summary_df["segment"] == "after_shift"].set_index("algorithm")

    algorithms = ["FIFO", "LRU", "Learned", "Optimal"]
    df_before = df_before.reindex(algorithms)
    df_after = df_after.reindex(algorithms)

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    rects1 = ax.bar(x - width/2, df_before["hit_ratio"] * 100, width, label="Phase 1 (Locality-Heavy)", 
                    color="#2ECC71", edgecolor="#27AE60", alpha=0.9, linewidth=1)
    rects2 = ax.bar(x + width/2, df_after["hit_ratio"] * 100, width, label="Phase 2 (Random / Bursty)", 
                    color="#F39C12", edgecolor="#D68910", alpha=0.9, linewidth=1)

    ax.set_ylabel("Hit Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_title("Hit Ratio Comparison Before vs. After Workload Shift", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="#F8F9F9", edgecolor="#BDC3C7", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Attach value labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_overall_comparison(summary_df: pd.DataFrame, output_path: str) -> None:
    """Generate overall summary comparison figure."""
    df_overall = summary_df[summary_df["segment"] == "overall"].set_index("algorithm")
    algorithms = ["FIFO", "LRU", "Learned", "Optimal"]
    df_overall = df_overall.reindex(algorithms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    colors = [PALETTE.get(algo, "#7F8C8D") for algo in algorithms]

    # Bar chart 1: Overall Hit Ratio
    bars1 = ax1.bar(algorithms, df_overall["hit_ratio"] * 100, color=colors, edgecolor="#2C3E50", alpha=0.9)
    ax1.set_ylabel("Overall Hit Ratio (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Overall Hit Ratio", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")

    # Bar chart 2: Total Page Faults
    bars2 = ax2.bar(algorithms, df_overall["page_faults"], color=colors, edgecolor="#2C3E50", alpha=0.9)
    ax2.set_ylabel("Total Page Faults", fontsize=11, fontweight="bold")
    ax2.set_title("Total Page Faults", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight="bold")

    fig.suptitle("Overall Algorithm Performance Summary", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_workload_trace(trace: List[int], shift_index: int, output_path: str) -> None:
    """Generate scatter plot showing page reference behavior across steps."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    steps = list(range(len(trace)))
    
    ax.scatter(steps[:shift_index], trace[:shift_index], s=6, color="#2980B9", alpha=0.6, label="Phase 1: Locality-Heavy")
    ax.scatter(steps[shift_index:], trace[shift_index:], s=6, color="#E67E22", alpha=0.6, label="Phase 2: Random / Bursty")
    ax.axvline(x=shift_index, color="#C0392B", linestyle="--", linewidth=1.5, label=f"Workload Shift (Step {shift_index})")

    ax.set_xlabel("Reference Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Referenced Page ID", fontsize=11, fontweight="bold")
    ax.set_title("Page Access Pattern Across Workload Shift", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="#F8F9F9", edgecolor="#BDC3C7")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
