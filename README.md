# Praktikum 3: GPU Programming in Triton

A lab template for the AIA Master's course. You will implement an optimised matrix multiplication on a GPU using [Triton](https://github.com/triton-lang/triton), benchmark it against cuBLAS, and analyse the result with the SIMT execution model in mind.

> **Requires a CUDA GPU.** The notebook does not run on macOS or CPU-only machines. Use Google Colab, Kaggle, the EFI GPU cluster, or a local NVIDIA workstation.

---

## Learning objectives

After completing this lab you should be able to:

1. Explain the **SIMT execution model**: programs, warps, threads, and how a Triton kernel is mapped onto streaming multiprocessors.
2. **Implement** Triton kernels that progressively unlock GPU performance: scalar → vectorised → tiled.
3. **Benchmark and analyse** kernel performance — wall-clock time, GFLOPS, occupancy, arithmetic intensity, throttling.
4. **Profile** a kernel using `torch.profiler` and read a Perfetto/Nsight trace.

---

## Setup

### Option A — Google Colab

1. Open <https://colab.research.google.com>, **File → Upload notebook**, choose `l2-triton_matmul.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is fine).
3. Run the first cell — it installs Triton and prints `cuda:0`.

### Option B — Kaggle Notebooks

1. Create a new notebook on <https://kaggle.com>, **File → Upload Notebook**.
2. **Settings → Accelerator → GPU T4 x1** (or P100).
3. Run the first cell.

### Option C — Local CUDA workstation or EFI Cluster

```bash
git clone <this-repo-url> l2-gpu-triton
cd l2-gpu-triton
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab l2-triton_matmul.ipynb
```


## Tasks

| # | Topic | What you do |
|---|---|---|
| **1** | GPU characterisation | Print device properties and derive **peak FP32** from first principles. |
| **2** | Naive matmul analysis | Reason about the given `simple_matmul_kernel`: number of programs, loads/stores, FLOPs, arithmetic intensity. |
| **3** | Naive benchmark | Measure wall-time and GFLOPS at `N=1024`. Compare to peak. Explain the gap using `kernel_info`. |
| **4** | Vectorised kernel | **Implement** `naive_matmul_kernel` with a vectorised K-loop using `tl.arange` + `tl.sum`. Benchmark the speed-up. |
| **5** | Tiled kernel | **Implement** `simple_tiled_matmul_kernel` using `tl.dot` for block-level GEMM. Compute its arithmetic intensity. |
| **6** | Block-size sweep | Sweep `BLOCK ∈ {16, 32, 64, 128, ...}`. Find the value that fails (and why), and the value that's fastest. |
| **7** | Throughput sweep | Use `triton.testing.perf_report` to compare your tiled kernel against `torch.matmul` (cuBLAS) over `N ∈ [256, 4096]`. |
| **8** | Profiling | Use `torch.profiler` to capture a trace, open it in [Perfetto](https://ui.perfetto.dev), and answer the analysis questions. |

The full task descriptions, all sub-questions, and the deliverables live inside the notebook itself.

---

## Evaluation

Your submission will be assessed on:

- **Correctness** of the two kernels you implement (Tasks 4 & 5) — verified against `torch.matmul` with `torch.testing.assert_close`.
- **Quality of the analysis answers** in markdown cells — does the explanation follow from the SIMT execution model and the GPU's resource limits, or is it just numbers?
- **Benchmark plots and reasoning** — Tasks 3, 6, 7 expect numbers + an interpretation.
- **Profiler trace** — a screenshot or short paragraph describing what you saw.
- **Team presentation** — short walkthrough of your kernels, your benchmark plot, and one surprise/insight.

Submit the completed notebook (with all cells executed and outputs intact) plus any extra figures you reference.


## References

- [Triton tutorial — matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton language reference](https://triton-lang.org/main/python-api/triton.language.html)
- [NVIDIA CUDA C Programming Guide — execution model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- [Roofline performance model (Williams, Waterman, Patterson, 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785)
