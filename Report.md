# Lab Report – Matrix Multiplication on CPU
**Course:** AI Accelerators (AIA)
**Lab:** Praktikum 1
**Team members:** _(Timo (Notebook made for my PC specs), Miriam, Hamzeh)_
**Date:** _(18.04.2026)_

---

## Task 1 – System Characterisation

> Fill in the details of your machine. Use tools such as `lscpu`, `lstopo`, `/proc/cpuinfo`.

| Property | Value |
|---|---|
| CPU model | i7-1360p |
| Number of cores / threads |12/16|
| Base / Boost clock speed (GHz) |2.2GHz/5.0GHz|
| SIMD ISA (SSE4.2 / AVX2 / AVX-512 …) |No AVX|
| SIMD width (bits / floats per vector) |256 bits / 8 floats (for AVX2)|
| MAC units per core |2 FMA units (per P-Core)|
| L1 cache size (per core) |48 KB Data Cache (per P-Core)|
| L2 cache size (per core) |1.25 MB (per P-Core)|
| L3 cache size (shared)  |18,0MB|
| Peak theoretical throughput (GFLOP/s) |640 GFLOP/s (P-Cores only, max boost)|

**How did you calculate peak throughput?**

_(formula: cores × clock × SIMD_width × MACs_per_cycle)_
Calculation: 4 cores × 5.0 GHz × 8 floats × 2 FMA units × 2 operations = 640 GFLOP/s

---

## Task 2 – Loop Reordering

> Measure each loop ordering for matrix sizes 64, 128, 256, 512, 1024, 2048, 4096.

| Loop order | N=256 (GFLOP/s) | N=512 (GFLOP/s) | N=1024 (GFLOP/s) | Everything above takes too long |
|---|---|---|---|---|
| i-j-k (naive) | 3.77 | 3.95 | 0.97 | DNF |
| i-k-j | 22.84 | 20.96 | 24.75 | DNF |
| j-k-i | 27.25 | 30.88 | 24.90 | DNF |
| k-i-j | 31.29 | 27.90 | 26.52 | DNF |

**Best ordering found:** `k-i-j` (Closely followed by `i-k-j` and `j-k-i`)

**Why does this ordering perform best?**
The most critical factor for performance in C (which uses row-major memory layout) is having the `j` index in the innermost loop. This ensures a stride-1 memory access pattern for arrays `B` and `C`, allowing the CPU to load full cache lines perfectly. 

Both `i-k-j` and `k-i-j` share this optimal inner loop structure, which is why their performance is remarkably similar (around 24-31 GFLOP/s). On my specific hardware (Intel i7-1360P), `k-i-j` edges out `i-k-j` slightly, likely due to how the specific hardware prefetchers and the L2/L3 cache hierarchy handle the outer loop iterations. 

*(Note: The surprisingly high performance of `j-k-i` suggests that the modern GCC compiler recognized the sub-optimal access pattern and performed automatic loop-interchange optimizations in the background).*

---

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

| Flags added | N=1024 (GFLOP/s) | Speedup vs. baseline |
|---|---|---|
| -O3 only (baseline) | 26.46 | 1.00× |
| -O3 -march=native | 31.32 | 1.18× |
| -O3 -march=native -ffast-math | 21.60 | 0.82× |
| -O3 -march=native -ffast-math -funroll-loops | 29.94 | 1.13× |
| -O3 -march=native -ffast-math -fopenmp-simd | 35.51 | 1.34× |
| -O3 -march=native -ffast-math -funroll-loops -fopenmp-simd | 30.46  | 1.15× |

**Did you add any `#pragma` hints to the source? If yes, which ones?**
Yes, I added `#pragma GCC ivdep` just above the innermost `j` loop in the `i-k-j` ordering. This hint tells the compiler to ignore assumed vector dependencies, assuring it that memory accesses in the loop do not overlap. This allows the compiler to safely generate packed SIMD instructions (AVX2).

**What speedup did you achieve? Why?**
Overall, I achieved a **1.34× speedup** (from ~26 to ~35 GFLOP/s) purely through compiler optimizations. 
* `-march=native` provided a solid boost (+18%) by unlocking the specific AVX2 instruction set of my i7-1360P. 
* Interestingly, adding `-ffast-math` temporarily *decreased* performance. This is a common real-world anomaly where relaxing IEEE-754 math rules can sometimes cause the compiler to choose sub-optimal instruction scheduling or interferes with loop heuristics.
* `-fopenmp-simd` (combined with the `#pragma` hint) delivered the final massive boost, successfully forcing the CPU to process multiple floating-point operations in a single cycle.

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=512 (GFLOP/s) | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|---|
| 32  | 24.03 | 22.85 | DNF (Timeout) |
| 64  | 20.64 | 19.33 | DNF (Timeout) |
| 128 | 37.60 | 36.25 | DNF (Timeout) |
| 256 | 46.26 | 39.03 | DNF (Timeout) |

**Best tile size:** `256`

**Why does this tile size work best for your machine?**
The optimal tile size is closely tied to the CPU's cache sizes. My i7-1360P has a 1.25 MB L2 cache per P-Core. A tile size of 256x256 elements means working with sub-matrices that take up $256 \times 256 \times 4 \text{ bytes} \approx 256 \text{ KB}$ of memory. Keeping three such blocks (from matrices A, B, and C) in cache requires roughly $768 \text{ KB}$. This fits perfectly into the ultra-fast 1.25 MB L2 cache, minimizing expensive trips to the L3 cache or main memory, while the blocks are large enough to amortize the overhead of the additional nested loops.

*(Note regarding N=4096: The benchmark for N=4096 was ultimately aborted and excluded. Even after disabling the extremely slow naive implementation, the sheer size of the data structures caused the execution to take an unreasonably long time. Despite the tiling optimizations, the massive memory footprint likely led to severe L3 cache thrashing and RAM bottlenecking on this specific laptop hardware).*

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=1024 (GFLOP/s) | Speedup vs 1 Thread |
|---|---|---|
| 1 | 37.05 | 1.00× |
| 2 | 40.62 | 1.10× |
| 4 | 38.43 | 1.04× |
| 8 | 41.12 | 1.11× |
| 16 (max threads) | 41.62 | 1.12× |

**Does throughput scale linearly with threads? Why / why not?**
No, the throughput does not scale linearly at all. Increasing the workload from 1 to 16 threads only resulted in a marginal performance increase of about 12%. There are three main reasons for this behavior:

1. **The Memory Wall:** Even with loop tiling, the algorithm becomes severely memory-bound at higher thread counts. All cores are competing for the same limited memory bandwidth to fetch matrix data from RAM. Once the bandwidth is saturated, additional cores just wait idly for data (memory stall).
2. **Hybrid CPU Architecture & Clock Speeds:** My i7-1360P utilizes a hybrid architecture with 4 Performance-Cores (P-Cores) and 8 Efficiency-Cores (E-Cores). A single thread can run on a P-Core at a maximum boost clock of 5.0 GHz. When utilizing more threads (like 4 or 8), the workload spills over to the significantly slower E-Cores, dragging down the average computational speed.
3. **Thermal Throttling & Overhead:** Running all 16 hardware threads simultaneously generates massive heat in a laptop chassis. The CPU aggressively lowers the clock frequencies across all cores to prevent overheating. Additionally, the OpenMP thread creation and synchronization overhead eats into the small performance gains.

---

## Task 6 – Performance Analysis

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).
My implementation is heavily **memory-bound**. Matrix multiplication inherently has an arithmetic intensity of $O(N^3)$ operations for $O(N^2)$ data. While loop tiling increases the effective arithmetic intensity by keeping data in the cache, the achieved maximum performance of ~41.6 GFLOP/s is still far below the theoretical compute peak of ~640 GFLOP/s. This massive gap indicates that the CPU cores spend a significant amount of time stalled, waiting for data to arrive from the main memory via the limited memory bus bandwidth.

**Comparison vs. PyTorch (N=1024):**

| Implementation | GFLOP/s | % of PyTorch |
|---|---|---|
| Naive C | ~0.98 | 0.2% |
| Best optimised C | 41.62 | 9.6% |
| PyTorch (CPU) | 433.31 | 100.0% |

**What is the gap and why does it exist?**
The gap between our best C implementation and PyTorch is roughly 10.4×. This exists because PyTorch relies on highly tuned BLAS (Basic Linear Algebra Subprograms) libraries like Intel MKL. While our C compiler does a great job with auto-vectorization, BLAS libraries use hand-written Assembly code specifically tailored to the processor architecture. They implement complex strategies like sub-register blocking, continuous memory packing (to prevent TLB misses), dynamic thread scheduling, and utilize hardware prefetchers perfectly—optimizations that a general-purpose C compiler simply cannot automatically derive from nested loops.

---

## Task 7 – Key Takeaways

_Write 3–5 sentences summarising the most important lessons learned from this lab._

This lab demonstrated that raw mathematical operation counts are practically irrelevant if memory access patterns are ignored, simply reordering loops to respect row-major storage and spatial locality improved performance by an order of magnitude. Furthermore, modern CPU hardware (like AVX2 vector units) remains severely underutilized unless we explicitly guide the compiler using aggressive optimization flags (`-O3`, `-march=native`) and `#pragma` hints. We also learned that for large datasets, manual cache blocking (tiling) is essential to bridge the speed gap between fast L1/L2 caches and slow main memory. Finally, the multithreading experiments proved that scaling is rarely linear, for memory-bound tasks on hybrid architectures, the memory bandwidth wall quickly stifles the benefit of adding more CPU threads.

---

## Figures

> Place your performance plots (GFLOP/s vs. matrix size) in the `figures/` folder and reference them here.

![Performance comparison](figures/performance.png)
