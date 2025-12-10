
# GPU-Accelerated Matrix Multiplication & Image Convolution

## 🧠 Overview
This project explores **high-performance GPU computing** through **matrix multiplication** and **2D image convolution** using CUDA, cuBLAS, and Python integration via shared libraries.

I implemented and benchmarked multiple versions of matrix multiplication — from baseline CPU code to optimized CUDA kernels — and extended the work into **real-time image filtering** (blur and edge detection) accelerated through GPU parallelism.  

> 💡 The goal: demonstrate how **GPU memory hierarchies, shared memory tiling, and kernel-level optimization** can yield thousand-fold speedups over traditional CPU implementations.


## ⚙️ Key Features

| Component | Description | Key Gains |
|------------|-------------|-----------|
| **CPU Matrix Multiply (C)** | Sequential baseline for runtime comparison | Baseline |
| **Naïve CUDA Kernel** | One thread per output element | ~275× faster than CPU at N=512 |
| **Tiled Shared Memory Kernel** | Introduced block tiling and data reuse | ~1.6× faster than naïve CUDA |
| **cuBLAS Integration** | Leveraged NVIDIA’s optimized SGEMM routine | 7× faster than custom kernel at 2048×2048 |
| **CUDA Convolution** | GPU-accelerated blur and edge detection filters | Up to 130× faster than CPU convolution |
| **Python Shared Library (`libconvolve.so`)** | Exposed CUDA kernels to Python for rapid prototyping | Seamless GPU acceleration in NumPy-style workflows |


## 📊 Performance Summary

### 🧮 Matrix Multiplication

| Implementation | N=512 | N=1024 | N=2048 |
|----------------|-------|--------|--------|
| **Naïve CUDA** | 0.0012s | 0.0092s | 0.0753s |
| **CPU (C)** | 0.338s | 3.187s | 77.65s |
| **Tiled CUDA** | 0.00084s | 0.00589s | 0.0464s |
| **cuBLAS** | 0.00096s | 0.00163s | 0.00685s |

> ⚙️ **Result:** GPU achieved up to **10,000× speedup** over CPU.  
> Shared-memory tiling yielded ~1.6× faster runtime vs naïve CUDA kernels.


### 🖼️ Image Convolution

Performed 2D convolution on images up to **13,000×13,000 pixels** for blur and edge detection filters.

| Filter Size (N) | CPU | CUDA Binary | CUDA Python Lib |
|-----------------|-----|-------------|-----------------|
| **3×3** | 14.11s | 1.50s | 0.09s |
| **5×5** | 17.08s | 1.50s | 0.11s |
| **7×7** | 23.17s | 1.52s | 0.13s |

> 🧩 The Python wrapper version ran **>150× faster than CPU** and **~10× faster than native CUDA binaries**, thanks to efficient GPU memory handling and minimal host-device transfer overhead.


## 🧩 Implementation Details

### Matrix Multiplication Pipeline
1. **CPU Implementation:** Baseline C code using triple nested loops.  
2. **Naïve CUDA:** 1 thread = 1 output element; global memory reads per multiply.  
3. **Tiled CUDA:** Used 16×16 shared memory tiles to improve data locality.  
4. **cuBLAS:** Called `cublasSgemm()` for hardware-optimized GEMM.  

### Image Convolution
- Implemented custom **edge detection** and **blur** filters in C and CUDA.  
- Exposed both kernels through a **shared library (`libconvolve.so`)** using `extern "C"`.  
- Integrated GPU filtering into Python with `ctypes` and `NumPy`.  

## 🔬 Analysis & Learnings

- **GPU Parallelism Scales Exponentially:** Even at modest sizes, GPUs achieve 100×–10,000× speedups due to thousands of concurrent threads.  
- **Memory Optimization Dominates:** Tiling and shared memory reuse yield measurable gains beyond raw compute parallelism.  
- **cuBLAS Outperforms Custom Kernels:** Tensor cores, warp-level scheduling, and vectorized instructions give cuBLAS a clear advantage at scale.  
- **Python Integration Works:** Using `ctypes` and shared libraries allows lightweight ML pipelines to tap into CUDA performance without recompilation.  
- **Transfer Overhead Matters:** For small matrices or images, GPU copy time can dominate; efficient batching mitigates this.

## 🧠 Insights
> "Optimizing for GPUs isn’t just about raw FLOPs — it’s about *managing data flow, memory access, and parallel efficiency.*"

This project solidified my understanding of:
- CUDA memory hierarchy and thread/block design.  
- Trade-offs between compute intensity and data transfer overhead.  
- Extending C/CUDA performance into Python ML environments.


## 🛠️ Tech Stack
**Languages:** C, CUDA C++, Python  
**Libraries:** cuBLAS, NumPy, ctypes  
**Hardware:** NVIDIA Tesla T4 / V100  
**Platform:** Google Cloud Compute Engine (Ubuntu 20.04)

## 🧰 Repository Structure
```bash
gpu-matrix-convolution/
├── matrix_cpu.c              # CPU matrix multiplication
├── matrix_gpu.cu             # Naïve CUDA kernel
├── matrix_tiled.cu           # Optimized tiled kernel
├── matrix_cublas.cu          # cuBLAS implementation
├── convolve_cuda.cu          # CUDA convolution (blur/edge)
├── convolution_cpu_img.c     # CPU convolution baseline
├── libconvolve.so            # Shared library for Python
├── benchmark_all.py          # Python benchmarking script
└── benchmark_v2.sh           # Bash script for matrix benchmarks
```

## 📈 Future Work
- Add **multi-GPU scaling** via CUDA streams and unified memory.  
- Integrate **cuDNN** for more advanced convolution pipelines.  
- Explore **quantized kernels** and TensorRT deployment for inference acceleration.  
- Build a **Flask-based web demo** showcasing GPU-accelerated image filters in real time.

