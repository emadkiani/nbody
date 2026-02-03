# CUDA-Accelerated N-Body Simulation

## Overview
This project implements a **gravitational N-Body simulation** using both a **sequential CPU implementation** and **parallel GPU implementations using CUDA**. The goal is to demonstrate how massively parallel GPU architectures can dramatically accelerate computationally expensive problems that are impractical to run efficiently on CPUs alone. The project is designed for **educational purposes** in parallel programming, CUDA optimization, and performance analysis.

The project includes:

* A baseline CPU version
* A baseline GPU version (global memory)
* An optimized GPU version (shared memory, tuned block size)
* Performance measurement and validation
* CSV-based result logging and Python-based plotting

---

## Hardware and Software Requirements

### Hardware

* CPU: Any modern x86-64 CPU
* GPU: NVIDIA GPU with CUDA support

### Software

* Windows 11
* WSL2 (Ubuntu 20.04 or newer)
* NVIDIA GPU Driver (Windows)
* CUDA Toolkit 12.4
* GCC / G++
* Python 3 (for plotting)

### Tested System Info:
```
CPU: Intel Core i5-12500H
GPU: NVIDIA RTX 3050 4GB Laptop (CC 8.6)
CUDA: 12.4
OS: Windows 11 + WSL2 (Ubuntu 24.04 LTS)
```

---

## Installation Guide (Windows + WSL)

### Install WSL2

Open **PowerShell (Administrator)** and run:

```powershell
wsl --install
```

### Install NVIDIA Driver (Windows)

To have access GPU inside WSL, make sure you have latest NVIDIA GPU driver: [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

### Install CUDA Toolkit inside WSL

Inside Ubuntu (WSL terminal):

```bash
sudo apt update
sudo apt install -y build-essential
```

Install CUDA Toolkit 12.4:

```bash
sudo apt install -y cuda-toolkit-12-4
```

Verify installation:

```bash
nvcc --version
nvidia-smi
```


### Install Python Dependencies (for plots)

```bash
sudo apt install -y python3 python3-pip
pip3 install pandas matplotlib
```

---

## Project Structure

```text
nbody/
├── src/
│   ├── common.h
│   ├── nbody_cpu.cpp
│   ├── nbody_gpu_baseline.cu
│   ├── nbody_gpu_optimized.cu
│   └── nbody_gpu_shared.cu
├── plot_nbody.py
└── README.md
```

---

## Build Instructions

First of all remove old binaries and CSV:

```bash
rm -f results_nbody.csv nbody_cpu nbody_gpu_optimized speedup_vs_n.png time_vs_n.png
```

### Compile CPU Version

```bash
g++ -O3 -march=native src/nbody_cpu.cpp -o nbody_cpu
```

### Compile GPU Optimized Version

```bash
nvcc -O3 src/nbody_gpu_optimized.cu -o nbody_gpu_optimized
```

Run for multiple problem sizes:

```bash
./nbody_cpu 1024
./nbody_cpu 2048
./nbody_cpu 4096
./nbody_cpu 8192
./nbody_cpu 16384

./nbody_gpu_optimized 1024
./nbody_gpu_optimized 2048
./nbody_gpu_optimized 4096
./nbody_gpu_optimized 8192
./nbody_gpu_optimized 16384
```

Each run appends timing results to `results_nbody.csv`.

### Compile GPU Baseline Version

To test GPU Baseline or GPU Shared version, rebuild again but this time:

```bash
nvcc -O3 src/nbody_gpu_baseline.cu -o nbody_gpu_baseline
```
or

```bash
nvcc -O3 src/nbody_gpu_shared.cu -o nbody_gpu_shared
```

---

## Generating Plots

Run:

```bash
python3 plot_nbody.py
```

This generates:

* `time_vs_n.png` – CPU vs GPU execution time
* `speedup_vs_n.png` – GPU speedup over CPU

---

## Validation

Correctness is validated by comparing particle positions between CPU and GPU runs.
Results match within floating-point tolerance, confirming correct parallel execution.

---

## Optimization Techniques Used

* CUDA grid–block tuning (128 / 256 threads per block)
* Shared memory tiling
* Coalesced global memory access
* Reduced global memory bandwidth usage
* Minimal warp divergence
* CUDA event timing for accurate measurement

---

## Educational Purpose

This project demonstrates:

* Why GPUs outperform CPUs for highly parallel workloads
* How algorithm structure affects GPU efficiency
* Practical CUDA optimization strategies
* Performance scaling with problem size