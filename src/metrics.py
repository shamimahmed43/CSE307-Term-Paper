"""
metrics.py - Metric Calculation and Evaluation Summary

Calculates page faults, hits, hit ratios, and fault ratios segmented into:
- before_shift (first half)
- after_shift (second half)
- overall (total trace)

Enforces mathematical invariants (hits + faults == references).
"""

from typing import List, Dict, Any
import pandas as pd
from src.classical_policies import SimulationResult


def compute_segment_metrics(
    result: SimulationResult,
    shift_index: int
) -> List[Dict[str, Any]]:
    """
    Compute metrics broken down into before_shift, after_shift, and overall segments.
    """
    records = result.records
    total_refs = len(records)
    
    before_records = records[:shift_index]
    after_records = records[shift_index:]
    
    segments = [
        ("before_shift", before_records),
        ("after_shift", after_records),
        ("overall", records)
    ]
    
    rows = []
    for seg_name, seg_recs in segments:
        ref_count = len(seg_recs)
        hits = sum(1 for r in seg_recs if r.is_hit)
        faults = sum(1 for r in seg_recs if not r.is_hit)
        
        # Verify invariant
        assert hits + faults == ref_count, (
            f"Invariant violation in {result.algorithm_name} [{seg_name}]: "
            f"hits ({hits}) + faults ({faults}) != references ({ref_count})"
        )
        
        hit_ratio = hits / ref_count if ref_count > 0 else 0.0
        fault_ratio = faults / ref_count if ref_count > 0 else 0.0
        
        rows.append({
            "algorithm": result.algorithm_name,
            "segment": seg_name,
            "references": ref_count,
            "hits": hits,
            "page_faults": faults,
            "hit_ratio": round(hit_ratio, 4),
            "fault_ratio": round(fault_ratio, 4)
        })
        
    return rows


def generate_summary_table(
    results: List[SimulationResult],
    shift_index: int
) -> pd.DataFrame:
    """Generate a combined summary DataFrame from all algorithm simulation results."""
    all_rows = []
    for res in results:
        all_rows.extend(compute_segment_metrics(res, shift_index))
    return pd.DataFrame(all_rows)


def save_summary_metrics(df: pd.DataFrame, filepath: str) -> None:
    """Save the summary metrics dataframe to CSV."""
    df.to_csv(filepath, index=False)


def save_detailed_simulation_trace(result: SimulationResult, filepath: str) -> None:
    """Save step-by-step access log for an algorithm run."""
    data = []
    for r in result.records:
        data.append({
            "step": r.step,
            "page": r.page,
            "is_hit": r.is_hit,
            "evicted_page": r.evicted_page if r.evicted_page is not None else "",
            "resident_frames": ",".join(map(str, r.resident_pages)),
            "phase": r.phase
        })
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
