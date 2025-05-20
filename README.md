# GPU Kernel Programming with Triton

This repository contains a series of custom implementations and explorations of GPU kernel-level programming using [Triton](https://triton-lang.org/), focused on performance optimization and low-level memory access.

Each module builds upon the last, culminating in applied techniques for linear algebra, attention mechanisms, and cross-entropy loss computation — all optimized for GPU architectures (tested on RTX 40-series cards).

## 🌟 Key Highlights

- 🚀 **Memory-Efficient GPU Code**: Explicit memory movement and vectorized operations
- 📊 **Real-World Performance Benchmarking**: Tested on RTX A6000 and 40-series cards
- 🔍 **Detailed Inline Commentary**: Step-by-step explanation of memory scope, threading, and compute
- 🧠 **Mathematics + Systems Blend**: CUDA-level control without CUDA complexity

## 📁 Project Structure

| Folder                 | Description                                           |
|------------------------|-------------------------------------------------------|
| `01_syllabus_day`      | Intro and memory fundamentals                         |
| `02_GPU_architecture`  | Architecture-level GPU concepts                       |
| `03_cloud_GPU_setup`   | Notes/scripts for cloud-based Triton environment      |
| `04_vector_addition`   | First kernel + Triton basics                          |
| `05_fused_softmax`     | Custom softmax with shared memory                     |
| `06_matmul`            | Fast Matrix Multiplication from scratch              |
| `07_dropout`           | Dropout kernel with random mask generation           |
| `08_layernorm`         | LayerNorm implementation and fusion                   |
| `09_flash_attention`   | Efficient attention kernel with block-wise processing |
| `10_CEloss_project`    | End-to-end fused CE loss with memory profiling        |

## ⚙️ Environment

- Python ≥ 3.9
- Triton ≥ 2.1
- Nvidia GPU with compute capability ≥ 8.0 (e.g., A6000, RTX 40XX recommended)

Install dependencies:
```bash
pip install -r requirements.txt
