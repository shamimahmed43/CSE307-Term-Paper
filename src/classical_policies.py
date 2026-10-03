"""
classical_policies.py - Classical Page Replacement Algorithms

Implements standard page replacement algorithms from scratch:
- FIFO (First-In, First-Out): Evicts the page resident the longest.
- LRU (Least Recently Used): Evicts the page accessed least recently.
- Optimal (Belady's MIN): Evicts the page whose next access is furthest in the future.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class AccessRecord:
    step: int
    page: int
    is_hit: bool
    resident_pages: List[int]
    evicted_page: Optional[int]
    phase: str


@dataclass
class SimulationResult:
    algorithm_name: str
    num_frames: int
    trace_length: int
    shift_index: int
    total_hits: int
    total_faults: int
    records: List[AccessRecord] = field(default_factory=list)

    @property
    def hit_ratio(self) -> float:
        return self.total_hits / self.trace_length if self.trace_length > 0 else 0.0

    @property
    def fault_ratio(self) -> float:
        return self.total_faults / self.trace_length if self.trace_length > 0 else 0.0


def run_fifo(trace: List[int], num_frames: int, shift_index: int = 1000) -> SimulationResult:
    """Simulate First-In First-Out (FIFO) page replacement."""
    frames: List[int] = []  # maintains insertion order
    records: List[AccessRecord] = []
    hits = 0
    faults = 0

    for step, page in enumerate(trace):
        phase = "before_shift" if step < shift_index else "after_shift"
        if page in frames:
            hits += 1
            records.append(AccessRecord(
                step=step, page=page, is_hit=True,
                resident_pages=list(frames), evicted_page=None, phase=phase
            ))
        else:
            faults += 1
            evicted = None
            if len(frames) < num_frames:
                frames.append(page)
            else:
                evicted = frames.pop(0)  # FIFO: remove oldest loaded
                frames.append(page)
            records.append(AccessRecord(
                step=step, page=page, is_hit=False,
                resident_pages=list(frames), evicted_page=evicted, phase=phase
            ))

    return SimulationResult(
        algorithm_name="FIFO",
        num_frames=num_frames,
        trace_length=len(trace),
        shift_index=shift_index,
        total_hits=hits,
        total_faults=faults,
        records=records
    )


def run_lru(trace: List[int], num_frames: int, shift_index: int = 1000) -> SimulationResult:
    """Simulate Least Recently Used (LRU) page replacement."""
    frames: Set[int] = set()
    last_accessed: Dict[int, int] = {}
    records: List[AccessRecord] = []
    hits = 0
    faults = 0

    for step, page in enumerate(trace):
        phase = "before_shift" if step < shift_index else "after_shift"
        if page in frames:
            hits += 1
            last_accessed[page] = step
            records.append(AccessRecord(
                step=step, page=page, is_hit=True,
                resident_pages=list(frames), evicted_page=None, phase=phase
            ))
        else:
            faults += 1
            evicted = None
            if len(frames) < num_frames:
                frames.add(page)
                last_accessed[page] = step
            else:
                # Evict resident page with minimum last_accessed step
                victim = min(frames, key=lambda p: (last_accessed.get(p, -1), p))
                frames.remove(victim)
                frames.add(page)
                last_accessed[page] = step
                evicted = victim
            records.append(AccessRecord(
                step=step, page=page, is_hit=False,
                resident_pages=list(frames), evicted_page=evicted, phase=phase
            ))

    return SimulationResult(
        algorithm_name="LRU",
        num_frames=num_frames,
        trace_length=len(trace),
        shift_index=shift_index,
        total_hits=hits,
        total_faults=faults,
        records=records
    )


def run_optimal(trace: List[int], num_frames: int, shift_index: int = 1000) -> SimulationResult:
    """
    Simulate Optimal / Belady's MIN page replacement.
    Uses future knowledge offline to evict the page accessed furthest in the future.
    """
    frames: Set[int] = set()
    records: List[AccessRecord] = []
    hits = 0
    faults = 0
    trace_len = len(trace)

    # Pre-build next occurrence lookup for efficiency
    # page -> list of step indices where page is requested
    occurrences: Dict[int, List[int]] = {}
    for step, page in enumerate(trace):
        if page not in occurrences:
            occurrences[page] = []
        occurrences[page].append(step)

    # Pointer for each page indicating next upcoming index in occurrences[page]
    occ_idx: Dict[int, int] = {p: 0 for p in occurrences}

    for step, page in enumerate(trace):
        phase = "before_shift" if step < shift_index else "after_shift"
        # Advance occurrence pointer for current page
        occ_idx[page] += 1

        if page in frames:
            hits += 1
            records.append(AccessRecord(
                step=step, page=page, is_hit=True,
                resident_pages=list(frames), evicted_page=None, phase=phase
            ))
        else:
            faults += 1
            evicted = None
            if len(frames) < num_frames:
                frames.add(page)
            else:
                # Find page in frames with largest next access index
                best_victim = None
                max_next_access = -1

                for resident in sorted(frames):  # deterministic tie-breaking
                    curr_ptr = occ_idx[resident]
                    occ_list = occurrences[resident]
                    if curr_ptr >= len(occ_list):
                        # Page is never accessed again in the trace
                        next_access = float("inf")
                    else:
                        next_access = occ_list[curr_ptr]

                    if next_access > max_next_access:
                        max_next_access = next_access
                        best_victim = resident

                victim = best_victim if best_victim is not None else min(frames)
                frames.remove(victim)
                frames.add(page)
                evicted = victim

            records.append(AccessRecord(
                step=step, page=page, is_hit=False,
                resident_pages=list(frames), evicted_page=evicted, phase=phase
            ))

    return SimulationResult(
        algorithm_name="Optimal",
        num_frames=num_frames,
        trace_length=len(trace),
        shift_index=shift_index,
        total_hits=hits,
        total_faults=faults,
        records=records
    )
