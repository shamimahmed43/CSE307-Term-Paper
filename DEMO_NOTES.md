# CSE-307 Demo Walkthrough & Viva Preparation Notes

**Project:** Learned Page Replacement under Dynamic Workload Distribution Shift  
**Duration:** 3–5 Minute Presentation Walkthrough + Viva Q&A  

---

## Part 1: Quick 3–5 Minute Presentation Walkthrough Script

### 1. Introduction & Motivation (30 seconds)
> "Good morning. My CSE-307 term paper investigates virtual memory page replacement algorithms under dynamic workload shifts. Specifically, I compared classical policies—FIFO and LRU—against Belády's theoretical Optimal policy and a lightweight Machine Learning model based on a Decision Tree."

### 2. The Replacement Policies & ML Design (60 seconds)
> "I implemented all algorithms from scratch:
> - **FIFO** replaces the oldest page in memory.
> - **LRU** replaces the page unaccessed for the longest time using temporal locality.
> - **Belády's Optimal (MIN)** looks into the future offline to evict the page accessed furthest ahead, establishing our theoretical upper bound.
> - **Learned Policy:** I trained a `DecisionTreeClassifier` on offline Optimal decisions from 5 separate training traces. At runtime, whenever frames are full during a page fault, it evaluates 4 past-only features for each resident page:
>   1. Recency (time since last reference)
>   2. Cumulative total access frequency
>   3. Sliding-window frequency (last 50 steps)
>   4. Resident age (time since loaded into memory).
> It evicts the page with the highest predicted eviction probability."

### 3. Experimental Setup & Workload Shift (45 seconds)
> "To prevent data leakage, training traces used separate random seeds from the held-out evaluation trace.
> The evaluation workload has 2,000 page accesses with a deliberate midpoint shift at step 1,000:
> - **Phase 1 (Steps 0–999):** Locality-heavy with a small 8-page working set and sequential scans.
> - **Phase 2 (Steps 1,000–1,999):** Random bursts switching across a 50-page universe."

### 4. Key Experimental Results (60 seconds)
> "Here is what our empirical results demonstrated ($K = 5$ frames):
> - **In Phase 1 (Locality Phase):** The Learned policy achieved a **59.60% hit ratio** (404 faults), clearly outperforming LRU at **53.20%** (468 faults) and FIFO at **48.10%** (519 faults). It combined recency with frequency to avoid cache pollution during sequential scans.
> - **In Phase 2 (Workload Shift):** The static Learned policy dropped to **13.30% hit ratio** (867 faults), underperforming LRU (**21.60%**) and FIFO (**21.50%**).
> - **Overall:** LRU achieved 37.40% hit ratio, Learned achieved 36.45%, FIFO achieved 34.80%, while Optimal reached 57.00%."

### 5. Conclusion & Core Takeaway (30 seconds)
> "The takeaway is that the learned policy performed well in the structured first phase but was not robust to the later random/bursty phase. Because the tree is fixed after offline training, a practical design would need online adaptation, periodic retraining, or a hybrid fallback such as LRU."

---

## Part 2: 10 Likely Viva Questions & Model Answers

### Q1: What is Belády's Optimal algorithm, and why can't real operating systems use it?
**Answer:** Belády's MIN algorithm evicts the page whose next access is furthest in the future. It cannot be used in real OS kernels because it requires omniscient knowledge of future memory references, which is impossible at runtime. It serves strictly as an offline theoretical benchmark.

### Q2: How did you ensure your Machine Learning model did not suffer from data leakage?
**Answer:** I strictly separated training traces (generated with seeds 100–104) from the held-out evaluation trace (seed 42). Furthermore, feature extraction during evaluation uses only historical information up to the current step ($t \le \text{current\_step}$), with zero lookahead into future references.

### Q3: What features did you feed into the Decision Tree classifier?
**Answer:** Four past-only features for each resident page:
1. Recency ($t - \text{last\_access}$)
2. Total access frequency so far
3. Sliding-window access frequency (last 50 steps)
4. Resident age ($t - \text{load\_time}$).

### Q4: How were the training labels generated for the ML model?
**Answer:** I simulated Belády's Optimal policy offline on the training traces. Whenever a page fault occurred in full frames, the page selected by Belády for eviction was labeled $y=1$ (positive eviction candidate), and all other resident candidates were labeled $y=0$.

### Q5: Why did the Learned model outperform LRU before the workload shift?
**Answer:** In Phase 1, the workload contained both high temporal locality and sequential scans. LRU only considers recency, so a sequential loop can quickly flush valuable working-set pages. The Decision Tree learned to combine frequency and resident age with recency, protecting frequently referenced hot pages from being prematurely evicted.

### Q6: Why did the Learned model perform worse than LRU after the workload shift?
**Answer:** In Phase 2, the access pattern became erratic and bursty across 50 pages. The experiment shows a large performance drop, but the aggregate metrics do not prove that cumulative frequency alone caused it. One plausible reason is state-distribution mismatch: training labels come from resident states produced by the Optimal oracle, while the learned policy creates its own resident states during evaluation. Earlier prediction errors can therefore change later candidate sets. The fixed model also does not retrain at the phase boundary.

### Q7: What is Belády's Anomaly, and which algorithm can suffer from it?
**Answer:** Belády's Anomaly is the phenomenon where increasing physical frame allocation causes the number of page faults to increase rather than decrease. FIFO can suffer from Belády's Anomaly because it is not a stack-based algorithm, whereas LRU and Optimal are stack algorithms and provably immune to it.

### Q8: How did you implement deterministic tie-breaking in your learned policy?
**Answer:** If multiple resident pages yield the exact same predicted eviction probability from the Decision Tree, the policy falls back to the page with the highest recency value (classical LRU tie-breaker), and if still tied, the smallest page ID.

### Q9: How did you verify the correctness and mathematical integrity of your simulation?
**Answer:** I implemented automated unit tests covering textbook traces for FIFO, LRU, and Optimal, ensured frame capacity bounds are strictly enforced at every step, and verified the invariant $\text{hits} + \text{page\_faults} = \text{total\_references}$ across all algorithms and segments.

### Q10: What is the practical overhead of learned page replacement in a real kernel?
**Answer:** A learned policy adds feature-maintenance and model-inference overhead compared with a simple heuristic. This project measures replacement quality, not real kernel latency, so I do not claim a specific microsecond cost. A production design would need to benchmark that overhead and may use a hybrid or approximate policy if inference is too expensive.
