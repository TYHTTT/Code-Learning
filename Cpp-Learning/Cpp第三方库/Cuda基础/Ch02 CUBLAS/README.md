# cuBLAS

全称为 CUDA Basic Linear Algebra Subprograms，是一个 GPU 加速库，用于加速 AI 和 HPC (高性能计算) 应用。它包含多种 API 扩展，提供可直接替换使用的行业标准 BLAS API 和支持操作融合的 GEMM (通用矩阵乘法) API，并针对 NVIDIA GPU 进行了高度优化。

# cuBLAS-Lt
- cuBLASLt (cuda BLAS Lightweight) 是 cuBLAS 库的扩展，提供更灵活的 API，主要用于提升深度学习模型等特定工作负载的性能。几乎所有数据类型和 API 调用都围绕矩阵乘法展开。

- 当一个问题无法通过单个内核完成时，cuBLASLt 会尝试将其分解为多个子问题，并对每个子问题运行内核来求解。

- fp16/fp8/int8 就是在这里发挥作用的。

# cuBLAS-Xt
cuBLAS-Xt (cuda BLAS Extended) 用于主机 + GPU 协同求解(速度慢得多)。

由于需要在主板上的 DRAM 和 GPU 的 VRAM 之间传输数据，会遇到内存带宽瓶颈，计算速度也就比所有操作都在芯片内完成时慢。

cuBLASXt 是 cuBLAS 的扩展，支持多 GPU。主要特性包括：

- 多 GPU： 能够在多个 GPU 上执行 BLAS 运算，实现 GPU 扩展，并有望显著提升大规模数据集的处理性能。

- 线程安全： 采用线程安全设计，支持在不同 GPU 上并发执行多个 BLAS 运算。

- 适用于能够通过将工作负载分配到多个 GPU 上而受益的大规模计算。

- 对于数据规模超出 GPU 显存容量的大规模线性代数计算，选择 XT。

# cuBLASDx

cuBLASDx (Device Extensions) 是一个设备端 API 扩展，用于在 CUDA 内核内部执行 BLAS 计算。通过融合数值运算，可以降低延迟，并进一步提升应用程序的性能。例如：

- 普通 cuBLAS：在 CPU 端的 C++ 代码里调用 `cublasSgemm(...)`，由 cuBLAS 启动 GPU 内核执行矩阵乘法。

- cuBLASDx：你自己启动 GPU 内核，内核中的 GPU 线程通过 cuBLASDx 协作完成矩阵乘法，再继续执行你写的其他计算。

# CUTLASS
全称为 CUDA Templates for Linear Algebra Subroutines

cuBLAS 及其变体由主机端调用，而 cuBLAS-DX 提供的功能在文档和优化方面都比较欠缺。

矩阵乘法是深度学习中最重要的运算，而 cuBLAS 对操作融合的支持用起来比较困难，CUTLASS 则支持实现这种融合。例如神经网络中的：

$$
Y=\operatorname{ReLU}(XW+b)
$$

- 分开执行：调用 cuBLAS 计算 $XW$，把结果写入显存，再启动后续内核，读取结果、加上偏置 $b$、执行 ReLU。
- 融合执行：用 CUTLASS 构建一个内核，在矩阵乘法结果写回显存前，直接完成加偏置和 ReLU，最后写出 $Y$。

顺便说明，Flash Attention 使用的是经过优化的 CUDA 内核，未使用 CUTLASS。