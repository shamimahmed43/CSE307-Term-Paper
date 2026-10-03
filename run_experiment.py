"""
run_experiment.py - Primary Experiment Driver for CSE-307 Term Paper

Executes the complete page replacement benchmark:
1. Generates separate training workloads and trains the Decision Tree learned policy offline.
2. Generates a held-out evaluation workload with a distinct midpoint shift (Locality -> Random/Bursty).
3. Evaluates FIFO, LRU, Optimal (Belady), and Learned policy on the exact same evaluation trace.
4. Computes segmented performance metrics (before shift, after shift, overall).
5. Validates correctness invariants.
6. Saves raw traces, summary CSVs, metadata, and generates publication-grade figures.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
import pandas as pd

# Add workspace to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.workload import generate_workload_trace, save_trace_to_csv
from src.classical_policies import run_fifo, run_lru, run_optimal, SimulationResult
from src.learned_policy import train_learned_policy, run_learned_policy
from src.metrics import generate_summary_table, save_summary_metrics, save_detailed_simulation_trace
from src.plots import (
    plot_page_faults_before_after,
    plot_hit_ratio_before_after,
    plot_overall_comparison,
    plot_workload_trace
)
from src.utils import ensure_directories, save_metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSE-307 Learned Page Replacement Experiment Runner"
    )
    parser.add_argument("--frames", type=int, default=5, help="Number of physical page frames (default: 5)")
    parser.add_argument("--trace-length", type=int, default=2000, help="Length of evaluation trace (default: 2000)")
    parser.add_argument("--shift-index", type=int, default=1000, help="Index of workload shift (default: 1000)")
    parser.add_argument("--universe-size", type=int, default=50, help="Total distinct page universe size (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for held-out evaluation trace (default: 42)")
    parser.add_argument("--num-train-traces", type=int, default=5, help="Number of distinct training traces (default: 5)")
    parser.add_argument("--train-seed-start", type=int, default=100, help="Starting seed for training traces (default: 100)")
    parser.add_argument("--max-depth", type=int, default=4, help="Max depth of Decision Tree (default: 4)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSV and logs (default: results)")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Directory for generated plots (default: figures)")
    parser.add_argument("--sensitivity", action="store_true", help="Run optional frame count sensitivity analysis (3, 5, 7)")
    return parser.parse_args()


def print_header(args: argparse.Namespace) -> None:
    print("=" * 78)
    print(" CSE-307: OPERATING SYSTEMS TERM PAPER EXPERIMENT")
    print(" Track 1: Learned Page Replacement under Dynamic Workload Distribution Shift")
    print("=" * 78)
    print(f" Frame Count:            {args.frames}")
    print(f" Evaluation Trace Len:   {args.trace_length} (Shift at index: {args.shift_index})")
    print(f" Page Universe Size:     {args.universe_size}")
    print(f" Evaluation Seed:        {args.seed}")
    print(f" Training Traces:        {args.num_train_traces} (Seeds: {args.train_seed_start}..{args.train_seed_start + args.num_train_traces - 1})")
    print(f" Decision Tree MaxDepth: {args.max_depth}")
    print("=" * 78 + "\n")


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(" EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'Algorithm':<12} | {'Segment':<14} | {'References':<10} | {'Hits':<6} | {'Faults':<6} | {'Hit Ratio':<10} | {'Fault Ratio':<10}")
    print("-" * 78)
    for _, row in summary_df.iterrows():
        print(f"{row['algorithm']:<12} | {row['segment']:<14} | {row['references']:<10} | {row['hits']:<6} | {row['page_faults']:<6} | {row['hit_ratio'] * 100:>8.2f}% | {row['fault_ratio'] * 100:>8.2f}%")
    print("=" * 78 + "\n")


def run_main_experiment(args: argparse.Namespace) -> pd.DataFrame:
    ensure_directories(".")

    # 1. Generate separate training traces to avoid data leakage
    print("[1/6] Generating training workloads & extracting oracle features...")
    training_traces = []
    for t_idx in range(args.num_train_traces):
        t_seed = args.train_seed_start + t_idx
        t_trace, _ = generate_workload_trace(
            trace_length=args.trace_length,
            shift_index=args.shift_index,
            universe_size=args.universe_size,
            seed=t_seed
        )
        training_traces.append(t_trace)

    # 2. Train Decision Tree offline on Belady oracle decisions
    print("[2/6] Training Decision Tree policy on offline Optimal oracle decisions...")
    model = train_learned_policy(
        training_traces=training_traces,
        num_frames=args.frames,
        max_depth=args.max_depth,
        random_state=args.seed
    )
    print(f"      Decision Tree trained successfully. (Max Depth: {args.max_depth})")

    # 3. Generate held-out evaluation trace
    print("[3/6] Generating held-out evaluation trace with midpoint workload shift...")
    eval_trace, eval_metadata = generate_workload_trace(
        trace_length=args.trace_length,
        shift_index=args.shift_index,
        universe_size=args.universe_size,
        seed=args.seed
    )
    
    # Save raw evaluation trace and metadata
    eval_trace_path = os.path.join(args.output_dir, "raw", "evaluation_trace.csv")
    save_trace_to_csv(eval_trace, eval_trace_path, shift_index=args.shift_index)
    
    metadata = {
        "experiment_parameters": vars(args),
        "workload_metadata": eval_metadata
    }
    save_metadata(metadata, os.path.join(args.output_dir, "raw", "experiment_metadata.json"))

    # 4. Simulate all algorithms on the EXACT same held-out trace
    print("[4/6] Executing page replacement algorithms on held-out evaluation trace...")
    res_fifo = run_fifo(eval_trace, args.frames, shift_index=args.shift_index)
    res_lru = run_lru(eval_trace, args.frames, shift_index=args.shift_index)
    res_opt = run_optimal(eval_trace, args.frames, shift_index=args.shift_index)
    res_learned = run_learned_policy(eval_trace, args.frames, model, shift_index=args.shift_index)

    results: List[SimulationResult] = [res_fifo, res_lru, res_learned, res_opt]

    # Save detailed access logs
    for res in results:
        detailed_path = os.path.join(args.output_dir, "raw", f"{res.algorithm_name.lower()}_detailed_trace.csv")
        save_detailed_simulation_trace(res, detailed_path)

    # 5. Compute metrics & save summary CSV
    print("[5/6] Computing metrics and verifying mathematical invariants...")
    summary_df = generate_summary_table(results, args.shift_index)
    summary_csv_path = os.path.join(args.output_dir, "summary_metrics.csv")
    save_summary_metrics(summary_df, summary_csv_path)

    # 6. Generate figures
    print("[6/6] Generating publication-quality figures...")
    plot_page_faults_before_after(summary_df, os.path.join(args.figures_dir, "page_faults_before_after.png"))
    plot_hit_ratio_before_after(summary_df, os.path.join(args.figures_dir, "hit_ratio_before_after.png"))
    plot_overall_comparison(summary_df, os.path.join(args.figures_dir, "overall_comparison.png"))
    plot_workload_trace(eval_trace, args.shift_index, os.path.join(args.figures_dir, "workload_distribution.png"))

    print_summary_table(summary_df)

    # Optional Sensitivity Analysis
    if args.sensitivity:
        print("[Optional] Running frame sensitivity analysis (frames = [3, 5, 7, 10])...")
        sens_rows = []
        for f in [3, 5, 7, 10]:
            f_model = train_learned_policy(training_traces, num_frames=f, max_depth=args.max_depth, random_state=args.seed)
            f_fifo = run_fifo(eval_trace, f, shift_index=args.shift_index)
            f_lru = run_lru(eval_trace, f, shift_index=args.shift_index)
            f_opt = run_optimal(eval_trace, f, shift_index=args.shift_index)
            f_lrn = run_learned_policy(eval_trace, f, f_model, shift_index=args.shift_index)
            for r in [f_fifo, f_lru, f_lrn, f_opt]:
                sens_rows.append({
                    "frames": f,
                    "algorithm": r.algorithm_name,
                    "overall_faults": r.total_faults,
                    "overall_hit_ratio": round(r.hit_ratio, 4)
                })
        sens_df = pd.DataFrame(sens_rows)
        sens_csv_path = os.path.join(args.output_dir, "sensitivity_metrics.csv")
        sens_df.to_csv(sens_csv_path, index=False)
        print(f"      Sensitivity results saved to {sens_csv_path}")

    return summary_df


if __name__ == "__main__":
    args = parse_arguments()
    print_header(args)
    run_main_experiment(args)
