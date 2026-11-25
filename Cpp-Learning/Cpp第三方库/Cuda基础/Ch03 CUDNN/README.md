cuDNN (NVIDIA CUDA Deep Neural Network library) 是 NVIDIA 专门为深度神经网络打造的底层 GPU 加速库。

如果把 CUDA 看作底层并行计算平台，cuBLAS 负责通用的基础线性代数 (如通用矩阵乘法 GEMM) ，那么 cuDNN 就是专门负责深度学习核心算子的“特种加速库”。

cuDNN 封装了深度学习中最耗时的底层原语 (Primitives) ，并针对不同代际的 NVIDIA GPU 架构 (尤其是 Tensor Core 和 FP16/BF16/FP8 低精度计算) 做了极致优化：

- 卷积 (Convolution) ：支持前向和反向传播，内置了隐式 GEMM、Winograd、FFT 等多种算法，能在运行时自动选择最快的实现。
- 注意力机制 (Attention) ：针对 Transformer 架构优化的融合注意力算子 (类似 FlashAttention 的底层支持) 。
- 归一化与池化：BatchNorm、LayerNorm、Max/Average Pooling 等。
- 激活函数：ReLU、GELU、Sigmoid、Tanh 等。
- 图级算子融合 (Graph API / Fusion) ：现代 cuDNN (v8/v9+) 的一大核心能力，可以把连续的多个操作 (比如 `Conv + Bias + ReLU`) 自动融合成单个 CUDA Kernel 执行，大幅减少中间结果读写显存的开销。

绝大多数开发者不需要直接写 cuDNN 的 C/C++ 代码。当安装并运行 PyTorch、TensorFlow、JAX 或 TensorRT 时，这些框架在底层已经集成了 cuDNN——在 Python 里写一行 `nn.Conv2d`，底层实际去调度的就是 cuDNN 编译好的高性能 Kernel。

只有在开发自定义推理引擎、或者需要手写极端优化的 C++/CUDA 深度学习算子时，才会直接调用 cuDNN 的底层 API。