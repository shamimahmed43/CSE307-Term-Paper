"""
test_policies.py - Automated Unit & Correctness Tests

Verifies:
1. Classical algorithm correctness on known analytical reference traces.
2. Invariants: hits + faults == trace length.
3. Memory capacity bounds: frame count never exceeded.
4. Determinism: identical seed produces identical results.
5. Workload shift index integrity.
6. Theoretical sanity: Optimal faults <= FIFO faults and Optimal faults <= LRU faults.
"""

import unittest
from src.workload import generate_workload_trace
from src.classical_policies import run_fifo, run_lru, run_optimal
from src.learned_policy import train_learned_policy, run_learned_policy
from src.metrics import compute_segment_metrics


class TestPageReplacementPolicies(unittest.TestCase):

    def setUp(self):
        # Known standard reference trace from OS literature
        # References: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5
        self.known_trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        self.num_frames = 3

    def test_fifo_known_trace(self):
        """Test FIFO on standard textbook trace with 3 frames."""
        result = run_fifo(self.known_trace, self.num_frames, shift_index=6)
        self.assertEqual(result.total_faults, 9)
        self.assertEqual(result.total_hits, 3)
        self.assertEqual(result.total_faults + result.total_hits, len(self.known_trace))

    def test_lru_known_trace(self):
        """Test LRU on standard textbook trace with 3 frames."""
        result = run_lru(self.known_trace, self.num_frames, shift_index=6)
        self.assertEqual(result.total_faults, 10)
        self.assertEqual(result.total_hits, 2)
        self.assertEqual(result.total_faults + result.total_hits, len(self.known_trace))

    def test_optimal_known_trace(self):
        """Test Optimal (Belady) on standard textbook trace with 3 frames."""
        result = run_optimal(self.known_trace, self.num_frames, shift_index=6)
        self.assertEqual(result.total_faults, 7)
        self.assertEqual(result.total_hits, 5)
        self.assertEqual(result.total_faults + result.total_hits, len(self.known_trace))

    def test_frame_capacity_constraint(self):
        """Ensure no algorithm ever exceeds allocated physical frame limit."""
        trace, _ = generate_workload_trace(trace_length=500, shift_index=250, seed=101)
        for num_frames in [3, 5, 8]:
            res_fifo = run_fifo(trace, num_frames)
            res_lru = run_lru(trace, num_frames)
            res_opt = run_optimal(trace, num_frames)

            for res in [res_fifo, res_lru, res_opt]:
                for rec in res.records:
                    self.assertLessEqual(len(rec.resident_pages), num_frames)

    def test_hit_fault_invariant_and_shift_index(self):
        """Verify hits + faults == trace_length across all segments."""
        trace, _ = generate_workload_trace(trace_length=1000, shift_index=500, seed=202)
        shift_idx = 500
        res_lru = run_lru(trace, 5, shift_index=shift_idx)
        metrics = compute_segment_metrics(res_lru, shift_idx)

        for seg in metrics:
            self.assertEqual(seg["hits"] + seg["page_faults"], seg["references"])
            if seg["segment"] == "before_shift":
                self.assertEqual(seg["references"], shift_idx)
            elif seg["segment"] == "after_shift":
                self.assertEqual(seg["references"], len(trace) - shift_idx)
            elif seg["segment"] == "overall":
                self.assertEqual(seg["references"], len(trace))

    def test_determinism(self):
        """Verify identical seed generates bitwise identical trace and simulation outcomes."""
        trace1, meta1 = generate_workload_trace(trace_length=600, shift_index=300, seed=42)
        trace2, meta2 = generate_workload_trace(trace_length=600, shift_index=300, seed=42)
        self.assertEqual(trace1, trace2)

        res1 = run_fifo(trace1, 5)
        res2 = run_fifo(trace2, 5)
        self.assertEqual(res1.total_faults, res2.total_faults)
        self.assertEqual(res1.total_hits, res2.total_hits)

    def test_optimal_optimality_bound(self):
        """Sanity check: Belady Optimal should have <= page faults than FIFO and LRU."""
        trace, _ = generate_workload_trace(trace_length=1000, shift_index=500, seed=77)
        frames = 5
        res_fifo = run_fifo(trace, frames)
        res_lru = run_lru(trace, frames)
        res_opt = run_optimal(trace, frames)

        self.assertLessEqual(res_opt.total_faults, res_fifo.total_faults)
        self.assertLessEqual(res_opt.total_faults, res_lru.total_faults)


if __name__ == "__main__":
    unittest.main()
