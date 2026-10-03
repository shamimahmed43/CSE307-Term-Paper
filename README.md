# Learned Page Replacement under Dynamic Workload Distribution Shift

**Course:** CSE-307: Operating Systems (Section B)  
**Track:** Track 1 — Learned Page Replacement (Memory Management)  
**Author:** Md Shamim Ahmed Limon  
**Student ID:** 202414043  

---

## 1. Executive Summary & Research Question

Virtual memory management relies on page replacement algorithms to minimize the frequency of high-latency disk I/O operations caused by page faults. Classical heuristic algorithms like **FIFO** and **LRU** employ fixed assumptions about temporal and sequential locality. 

### Core Research Question
> *Can a lightweight, explainable Machine Learning model (Decision Tree) learn effective page replacement decisions from an offline Optimal oracle, outperforming classical LRU during locality-heavy phases, and how does it react when the workload abruptly undergoes a distribution shift to random and bursty access patterns?*

---

## 2. Theoretical Overview of Algorithms

- **FIFO (First-In, First-Out):** Replaces the page that has been in physical memory the longest, regardless of its access frequency or recency. Simple $O(1)$ queue implementation, but prone to evicting heavily accessed pages and vulnerable to Belády's Anomaly.
- **LRU (Least Recently Used):** Exploits the principle of temporal locality by evicting the page that has not been referenced for the longest duration of time ($\min \text{last\_access}$).
- **Optimal (Belády's MIN Algorithm):** The theoretical upper bound. Evicts the resident page whose next access is furthest in the future (or never accessed again). Unachievable in real-time operating systems because it requires omniscient future knowledge.
- **Learned Policy (Decision Tree Classifier):** Evaluates candidate resident pages during a page fault using strictly past and current historical features (recency, cumulative frequency, sliding-window frequency, resident age). Predicts eviction probability learned from offline Belády oracle decisions.

---

## 3. Supervised Oracle Training & Feature Engineering

### Avoiding Data Leakage
To prevent data contamination, model training and benchmark evaluation are strictly separated:
- **Training Traces:** 5 independent synthetic traces (random seeds 100, 101, 102, 103, 104).
- **Held-Out Evaluation Trace:** A completely distinct trace (random seed 42) with an explicit distribution shift.

### Training Label Generation (Offline Oracle)
During simulation of the training traces under the Belády MIN policy, whenever a page fault occurs and all frames are full:
1. The Belády oracle identifies the optimal victim page (furthest future access).
2. For all candidate resident pages currently in physical frames, past-only features are recorded.
3. The true optimal victim is assigned label $y = 1$; all other resident pages receive label $y = 0$.
4. An explainable `DecisionTreeClassifier(max_depth=4, class_weight='balanced')` is fitted to this offline dataset.

### Inference Features (Strictly Past-Only — NO Future Leakage)
1. `recency`: Steps elapsed since last access ($t - \text{last\_access}$).
2. `total_frequency`: Cumulative reference count so far for this page.
3. `window_frequency`: Reference count within the recent sliding window ($W = 50$ steps).
4. `resident_age`: Steps elapsed since the page was loaded into physical memory ($t - \text{load\_time}$).

**Deterministic Tie-Breaking:** If candidate resident pages produce equal eviction probabilities, the tie is broken first by highest recency (LRU fallback), then by smallest page ID.

---

## 4. Workload Shift Design

The evaluation workload contains $N = 2,000$ page references over a universe of 50 virtual pages, split at midpoint **$S = 1,000$**:

1. **Phase 1: Locality-Heavy & Sequential (Steps 0–999)**
   - 65% accesses sampled from a compact 8-page working set ($p \in [0, 7]$) with geometric repetition.
   - 20% sequential loop traversals.
   - 15% random background access.
2. **Phase 2: Random / Bursty (Steps 1,000–1,999)**
   - 45% short-lived burst clusters (6 randomly chosen pages) that periodically switch every 15–35 steps.
   - 55% uniform random references across the entire 50-page universe.

---

## 5. Summary of Real Experimental Results

The experiment was executed with $K = 5$ frames, trace length $N = 2000$, and shift index $S = 1000$.

| Algorithm | Phase / Segment | References | Hits | Page Faults | Hit Ratio | Fault Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | Before Shift (Phase 1) | 1,000 | 481 | 519 | 48.10% | 51.90% |
| | After Shift (Phase 2) | 1,000 | 215 | 785 | 21.50% | 78.50% |
| | **Overall** | **2,000** | **696** | **1,304** | **34.80%** | **65.20%** |
| **LRU** | Before Shift (Phase 1) | 1,000 | 532 | 468 | 53.20% | 46.80% |
| | After Shift (Phase 2) | 1,000 | 216 | 784 | 21.60% | 78.40% |
| | **Overall** | **2,000** | **748** | **1,252** | **37.40%** | **62.60%** |
| **Learned (Tree)** | Before Shift (Phase 1) | 1,000 | 596 | 404 | **59.60%** | **40.40%** |
| | After Shift (Phase 2) | 1,000 | 133 | 867 | 13.30% | 86.70% |
| | **Overall** | **2,000** | **729** | **1,271** | **36.45%** | **63.55%** |
| **Optimal (MIN)** | Before Shift (Phase 1) | 1,000 | 705 | 295 | **70.50%** | **29.50%** |
| | After Shift (Phase 2) | 1,000 | 435 | 565 | 43.50% | 56.50% |
| | **Overall** | **2,000** | **1,140** | **860** | **57.00%** | **43.00%** |

*All results strictly satisfy the invariant $\text{hits} + \text{faults} = \text{references}$.*

---

## 6. Key Scientific Findings & Discussion

1. **Learned Policy Performance in Structured Locality (Phase 1):**  
   The Decision Tree achieves a **59.60%** hit ratio, outperforming LRU (53.20%) and FIFO (48.10%) on this held-out trace. The result is consistent with the model benefiting from multiple past-only features rather than recency alone.
2. **Sensitivity to the Workload Phase Shift (Phase 2):**  
   When the workload becomes random and bursty, the static Learned policy drops to **13.30%** hit ratio (867 faults), underperforming both LRU (21.60%) and FIFO (21.50%). The aggregate metrics do not prove that any single feature caused this drop. A plausible contributor is state-distribution mismatch: the model is trained on candidate states generated by the Optimal oracle but, during evaluation, its own earlier eviction decisions create different resident-page states.
3. **Overall Result:**  
   LRU achieves a **37.40%** overall hit ratio, slightly above the Learned policy at **36.45%**. Optimal remains the offline upper-bound reference at **57.00%**.

---

## 7. Repository Structure

```
CSE307-Term-Paper/
│
├── README.md                      # Complete project documentation & results
├── requirements.txt               # Minimal Python package requirements
├── run_experiment.py              # Single-command experiment & benchmark driver
├── DEMO_NOTES.md                  # 3-5 minute presentation script & viva Q&A
├── .gitignore                     # Clean repository exclusions
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── workload.py                # Synthetic trace generator with distribution shift
│   ├── classical_policies.py      # FIFO, LRU, and Optimal MIN algorithms from scratch
│   ├── learned_policy.py          # Decision Tree classifier with Oracle training
│   ├── metrics.py                 # Segmented metric computer & invariant validator
│   ├── plots.py                   # Matplotlib publication-grade chart generator
│   └── utils.py                   # Directory & metadata persistence utilities
│
├── tests/
│   └── test_policies.py           # Automated unit tests for all policies
│
├── scripts/
│   ├── collect_environment.sh     # System hardware & OS specification profiler
│   └── run_resource_profile.sh    # VMware / Ubuntu CPU and Max-RSS profiler
│
├── results/
│   ├── raw/                       # Step-by-step traces, CSVs, and metadata JSON
│   ├── resource_profiles/         # Ubuntu / VMware hardware profiling logs
│   ├── summary_metrics.csv        # Segmented experimental metrics table
│   └── sensitivity_metrics.csv    # Frame capacity sensitivity data
│
├── figures/
│   ├── page_faults_before_after.png  # Before vs. After fault count comparison
│   ├── hit_ratio_before_after.png    # Before vs. After hit ratio comparison
│   ├── overall_comparison.png        # Overall algorithm performance summary
│   └── workload_distribution.png     # Scatter plot of page access shift
│
└── report/
    ├── main.tex                   # Academic 2-column LaTeX term paper draft
    └── references.bib             # Genuine BibTeX academic bibliography
```

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.9+ (or MSYS2 UCRT64 Python)
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Quick Start (One Command)
```bash
# Clone or enter directory
cd "CSE307-Term-Paper"

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m unittest tests/test_policies.py

# Run complete benchmark experiment
python run_experiment.py --sensitivity
```

### Optional Command Line Flags
```bash
python run_experiment.py --frames 5 --trace-length 2000 --shift-index 1000 --seed 42 --sensitivity
```

---

## 9. Secondary Ubuntu / VMware Profiling Instructions

To evaluate system execution characteristics in an Ubuntu virtual machine:
1. Boot Ubuntu VM inside VMware Workstation.
2. Clone this repository into the VM.
3. Collect system specs:
   ```bash
   bash scripts/collect_environment.sh
   ```
4. Run profile benchmarks under GNU `/usr/bin/time -v`:
   ```bash
   bash scripts/run_resource_profile.sh low     # (e.g. 1 vCPU, 1GB RAM)
   bash scripts/run_resource_profile.sh medium  # (e.g. 2 vCPU, 2GB RAM)
   bash scripts/run_resource_profile.sh high    # (e.g. 4 vCPU, 4GB RAM)
   ```
5. Measured logs with Maximum Resident Set Size (Max RSS) and CPU times will be stored in `results/resource_profiles/`.

**Important:** The scripts do **not** change VMware RAM/CPU settings by themselves. Manually configure the VM before each `low`, `medium`, or `high` run. No VM resource-profile measurements are included in this repository yet; do not claim resource conclusions until those runs are actually performed.

---

## 10. Known Limitations

1. **Synthetic vs. Production Traces:** While the synthetic workload models working-set locality and phase shifts clearly, real OS workloads have complex multi-threaded memory access patterns.
2. **Static Decision Tree vs. Online Retraining:** The learned model is trained offline. Real-world deployment would require lightweight online weight updating or hybrid fallback to LRU.
3. **Runtime Overhead:** Feature extraction and Decision Tree inference add CPU and memory overhead relative to simple classical heuristics. This project does not measure kernel-level page-replacement latency, so no microsecond-level overhead claim is made.

---

## 11. AI-Assistance Disclosure

AI coding assistance was used for implementation support, debugging, visualization scripting, documentation structuring, and LaTeX drafting. The experiment is reproducible from the included scripts and fixed seeds. The author is responsible for reviewing, understanding, and explaining the final code, results, and analysis.
