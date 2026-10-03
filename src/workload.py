"""
workload.py - Synthetic Workload Generator with Distribution Shift

Generates page reference sequences exhibiting a deliberate workload shift:
- First Half (Locality-heavy / Sequential):
    Strong temporal and spatial locality with a compact working set, 
    sequential loop traversals, and occasional background references.
- Second Half (Random / Bursty):
    Expanded page universe, erratic jump accesses, and short bursty clusters 
    that rapidly shift active sets.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def generate_workload_trace(
    trace_length: int = 2000,
    shift_index: int = 1000,
    universe_size: int = 50,
    locality_working_set_size: int = 8,
    seed: int = 42
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate a synthetic page-reference sequence with a distinct workload shift at shift_index.
    
    Parameters:
        trace_length: Total number of page references.
        shift_index: The index at which the workload shifts from locality to random/bursty.
        universe_size: Total distinct page IDs available.
        locality_working_set_size: Size of hot working set in the locality phase.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        trace: List of page references (integers).
        metadata: Dictionary describing the workload configuration.
    """
    rng = np.random.default_rng(seed)
    trace: List[int] = []

    # --- Phase 1: Locality-Heavy & Sequential (0 to shift_index - 1) ---
    # Hot working set pages: e.g., pages [0 .. locality_working_set_size - 1]
    hot_set = list(range(locality_working_set_size))
    seq_cursor = 0

    for i in range(shift_index):
        p_mode = rng.random()
        if p_mode < 0.65:
            # High temporal locality: sample from hot working set with geometric bias
            idx = int(rng.geometric(p=0.4)) - 1
            idx = idx % len(hot_set)
            page = hot_set[idx]
        elif p_mode < 0.85:
            # Sequential scan over working set / small window
            page = hot_set[seq_cursor % len(hot_set)]
            seq_cursor += 1
        else:
            # Occasional reference outside hot set
            page = int(rng.integers(locality_working_set_size, universe_size // 2))
        trace.append(int(page))

    # --- Phase 2: Random / Bursty (shift_index to trace_length - 1) ---
    # Page universe expands, bursts switch dynamically
    burst_set = list(rng.choice(universe_size, size=6, replace=False))
    burst_countdown = int(rng.integers(15, 35))

    for i in range(shift_index, trace_length):
        p_mode = rng.random()
        if burst_countdown <= 0:
            # Switch the hot burst cluster
            burst_set = list(rng.choice(universe_size, size=6, replace=False))
            burst_countdown = int(rng.integers(15, 35))
        burst_countdown -= 1

        if p_mode < 0.45:
            # Bursty access from current short-lived cluster
            page = int(rng.choice(burst_set))
        else:
            # Uniformly random access across the entire large page universe
            page = int(rng.integers(0, universe_size))
        trace.append(int(page))

    metadata = {
        "seed": seed,
        "trace_length": trace_length,
        "shift_index": shift_index,
        "universe_size": universe_size,
        "locality_working_set_size": locality_working_set_size,
        "phase1_description": f"Locality-heavy with small working set ({locality_working_set_size} pages) and sequential scans",
        "phase2_description": f"Random and shifting burst clusters across full page universe ({universe_size} pages)"
    }

    return trace, metadata


def save_trace_to_csv(trace: List[int], filepath: str, shift_index: int = 1000) -> None:
    """Save the generated trace to CSV with phase annotations."""
    df = pd.DataFrame({
        "step": list(range(len(trace))),
        "page_id": trace,
        "phase": ["before_shift" if i < shift_index else "after_shift" for i in range(len(trace))]
    })
    df.to_csv(filepath, index=False)
