"""
learned_policy.py - Machine Learning-Based Adaptive Page Replacement Policy

Implements a learned page replacement policy using DecisionTreeClassifier:
- Training:
    Uses offline Optimal/Belady oracle on separate training traces to label 
    which resident page was the ideal eviction candidate (y=1 vs y=0).
- Features:
    Computed strictly from PAST and CURRENT history (NO future leakage):
    1. recency: steps since last access (current_step - last_access_step)
    2. total_frequency: total number of times this page has been referenced so far
    3. window_frequency: number of accesses within recent sliding window (W=50)
    4. resident_age: steps since page was loaded into physical memory frames
- Inference:
    During evaluation on the held-out trace, when a fault occurs and frames are full,
    the model predicts the eviction probability for each candidate in memory.
    The page with highest eviction score is evicted.
    Deterministic tie-breaking is applied if scores match.
"""

from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import deque
import numpy as np

# Try importing scikit-learn; provide built-in fallback if environment lacks C-compiler for Py3.14
try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDT
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.classical_policies import SimulationResult, AccessRecord


FEATURE_NAMES = ["recency", "total_frequency", "window_frequency", "resident_age"]


class _DecisionTreeNode:
    """Internal node for lightweight pure-Python Decision Tree fallback."""
    def __init__(
        self,
        feature_index: int = -1,
        threshold: float = 0.0,
        left: Optional["_DecisionTreeNode"] = None,
        right: Optional["_DecisionTreeNode"] = None,
        prob_positive: float = 0.0,
        is_leaf: bool = False
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prob_positive = prob_positive
        self.is_leaf = is_leaf


class BuiltinDecisionTree:
    """
    Lightweight, self-contained binary Decision Tree classifier.
    Mimics scikit-learn's API (fit, predict_proba, classes_) using Gini impurity.
    """
    def __init__(self, max_depth: int = 4, class_weight: str = "balanced", random_state: int = 42):
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.root: Optional[_DecisionTreeNode] = None
        self.classes_ = np.array([0, 1])

    def _gini(self, y: np.ndarray, weights: np.ndarray) -> float:
        total_w = np.sum(weights)
        if total_w <= 0:
            return 0.0
        p1 = np.sum(weights[y == 1]) / total_w
        p0 = 1.0 - p1
        return 1.0 - (p0**2 + p1**2)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, depth: int) -> _DecisionTreeNode:
        n_samples = len(y)
        total_w = np.sum(weights)
        if total_w <= 0 or n_samples == 0:
            return _DecisionTreeNode(is_leaf=True, prob_positive=0.5)

        prob_pos = float(np.sum(weights[y == 1]) / total_w)

        # Base conditions: pure node or max depth
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or n_samples < 4:
            return _DecisionTreeNode(is_leaf=True, prob_positive=prob_pos)

        current_gini = self._gini(y, weights)
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        n_features = X.shape[1]
        for f in range(n_features):
            values = np.unique(X[:, f])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thresh in thresholds:
                left_mask = X[:, f] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                w_left = np.sum(weights[left_mask])
                w_right = np.sum(weights[right_mask])
                gini_split = (w_left / total_w) * self._gini(y[left_mask], weights[left_mask]) + \
                             (w_right / total_w) * self._gini(y[right_mask], weights[right_mask])
                gain = current_gini - gini_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        if best_gain <= 1e-6 or best_feat == -1:
            return _DecisionTreeNode(is_leaf=True, prob_positive=prob_pos)

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], weights[~left_mask], depth + 1)

        return _DecisionTreeNode(
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            prob_positive=prob_pos,
            is_leaf=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = len(y)
        if self.class_weight == "balanced":
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w_pos = (n_samples / (2.0 * max(n_pos, 1)))
            w_neg = (n_samples / (2.0 * max(n_neg, 1)))
            weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
        else:
            weights = np.ones(n_samples, dtype=np.float64)

        self.root = self._build_tree(X, y, weights, depth=0)
        return self

    def _predict_row_proba(self, row: np.ndarray, node: _DecisionTreeNode) -> float:
        if node.is_leaf or node.left is None or node.right is None:
            return node.prob_positive
        if row[node.feature_index] <= node.threshold:
            return self._predict_row_proba(row, node.left)
        else:
            return self._predict_row_proba(row, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Model not fitted.")
        probs_pos = np.array([self._predict_row_proba(X[i], self.root) for i in range(len(X))])
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])


def extract_candidate_features(
    page: int,
    current_step: int,
    last_accessed: Dict[int, int],
    total_freq: Dict[int, int],
    recent_history_window: deque,
    load_time: Dict[int, int]
) -> List[float]:
    """
    Extract past-only features for a resident page candidate.
    Strictly forbids looking at any future steps in the trace.
    """
    recency = float(current_step - last_accessed.get(page, current_step))
    tot_freq = float(total_freq.get(page, 0))
    win_freq = float(recent_history_window.count(page))
    res_age = float(current_step - load_time.get(page, current_step))

    return [recency, tot_freq, win_freq, res_age]


def generate_training_data_from_trace(
    trace: List[int],
    num_frames: int,
    window_size: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate trace using Optimal oracle to generate (features, label) pairs.
    Label is 1 for the optimal victim chosen by Belady, and 0 for non-victims.
    """
    frames: Set[int] = set()
    last_accessed: Dict[int, int] = {}
    load_time: Dict[int, int] = {}
    total_freq: Dict[int, int] = {}
    recent_window: deque = deque(maxlen=window_size)

    occurrences: Dict[int, List[int]] = {}
    for step, p in enumerate(trace):
        if p not in occurrences:
            occurrences[p] = []
        occurrences[p].append(step)
    occ_idx: Dict[int, int] = {p: 0 for p in occurrences}

    X_list: List[List[float]] = []
    y_list: List[int] = []

    for step, page in enumerate(trace):
        occ_idx[page] += 1
        total_freq[page] = total_freq.get(page, 0) + 1
        recent_window.append(page)

        if page in frames:
            last_accessed[page] = step
        else:
            if len(frames) < num_frames:
                frames.add(page)
                last_accessed[page] = step
                load_time[page] = step
            else:
                # Frames are full: eviction event!
                # 1. Determine optimal victim using Belady oracle
                best_victim = None
                max_next_access = -1

                for resident in sorted(frames):
                    curr_ptr = occ_idx[resident]
                    occ_list = occurrences[resident]
                    next_access = occ_list[curr_ptr] if curr_ptr < len(occ_list) else float("inf")
                    if next_access > max_next_access:
                        max_next_access = next_access
                        best_victim = resident

                optimal_victim = best_victim if best_victim is not None else min(frames)

                # 2. Extract past-only features for all candidate resident pages
                for resident in sorted(frames):
                    feat = extract_candidate_features(
                        page=resident,
                        current_step=step,
                        last_accessed=last_accessed,
                        total_freq=total_freq,
                        recent_history_window=recent_window,
                        load_time=load_time
                    )
                    X_list.append(feat)
                    y_list.append(1 if resident == optimal_victim else 0)

                # Perform oracle eviction to continue simulation
                frames.remove(optimal_victim)
                load_time.pop(optimal_victim, None)
                frames.add(page)
                last_accessed[page] = step
                load_time[page] = step

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, 4), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty((0,), dtype=np.int32)
    return X, y


def train_learned_policy(
    training_traces: List[List[int]],
    num_frames: int,
    max_depth: int = 4,
    random_state: int = 42
) -> Union[Any, BuiltinDecisionTree]:
    """
    Train a DecisionTreeClassifier model across multiple training traces.
    """
    all_X = []
    all_y = []

    for trace in training_traces:
        X_t, y_t = generate_training_data_from_trace(trace, num_frames)
        if len(X_t) > 0:
            all_X.append(X_t)
            all_y.append(y_t)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    if HAS_SKLEARN:
        clf = SklearnDT(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        clf = BuiltinDecisionTree(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state
        )
    clf.fit(X_train, y_train)
    return clf


def run_learned_policy(
    trace: List[int],
    num_frames: int,
    model: Any,
    shift_index: int = 1000,
    window_size: int = 50
) -> SimulationResult:
    """
    Execute learned page replacement policy on a held-out trace.
    Inference strictly uses historical and state data available at that step.
    """
    frames: Set[int] = set()
    last_accessed: Dict[int, int] = {}
    load_time: Dict[int, int] = {}
    total_freq: Dict[int, int] = {}
    recent_window: deque = deque(maxlen=window_size)
    records: List[AccessRecord] = []
    hits = 0
    faults = 0

    for step, page in enumerate(trace):
        phase = "before_shift" if step < shift_index else "after_shift"
        total_freq[page] = total_freq.get(page, 0) + 1
        recent_window.append(page)

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
                load_time[page] = step
            else:
                # Frames full: evaluate all resident candidates with Decision Tree
                candidate_list = sorted(frames)
                candidate_features = [
                    extract_candidate_features(
                        page=p,
                        current_step=step,
                        last_accessed=last_accessed,
                        total_freq=total_freq,
                        recent_history_window=recent_window,
                        load_time=load_time
                    )
                    for p in candidate_list
                ]

                # Predict probability of class 1 (victim)
                X_cand = np.array(candidate_features, dtype=np.float32)
                classes = list(model.classes_)
                if 1 in classes:
                    col_idx = classes.index(1)
                    probs = model.predict_proba(X_cand)[:, col_idx]
                else:
                    probs = np.zeros(len(candidate_list), dtype=np.float32)

                # Deterministic tie-breaking:
                # 1. Primary: highest predicted eviction probability
                # 2. Secondary: highest recency (steps since last access)
                # 3. Tertiary: lowest page ID
                best_score = -1.0
                best_victim = candidate_list[0]
                best_recency = -1.0

                for p, prob, feat in zip(candidate_list, probs, candidate_features):
                    recency_val = feat[0]
                    if (prob > best_score) or (
                        abs(prob - best_score) < 1e-6 and recency_val > best_recency
                    ):
                        best_score = prob
                        best_recency = recency_val
                        best_victim = p

                victim = best_victim
                frames.remove(victim)
                load_time.pop(victim, None)

                frames.add(page)
                last_accessed[page] = step
                load_time[page] = step
                evicted = victim

            records.append(AccessRecord(
                step=step, page=page, is_hit=False,
                resident_pages=list(frames), evicted_page=evicted, phase=phase
            ))

    return SimulationResult(
        algorithm_name="Learned",
        num_frames=num_frames,
        trace_length=len(trace),
        shift_index=shift_index,
        total_hits=hits,
        total_faults=faults,
        records=records
    )
