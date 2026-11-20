在 CUDA 及其核心加速库 (如 cuBLAS、cuBLASLt、cuDNN、cuSPARSE、cuSOLVER 等) 中：

- 句柄
- 矩阵乘法操作描述符
- 矩阵布局描述符
- 偏好描述符
- WorkSpace
- Stream

这六个概念共同构成了“描述任务 → 寻找最优算子 → 异步执行”的标准编程模型。

将它们串联起来看，一次典型的现代 CUDA 库调用流程就是：

在 **Handle** 的上下文中，用 **操作描述符** 和 **布局描述符** 定义好计算任务，结合 **偏好描述符** 筛选出最优算法，最后提供一块 **WorkSpace** 临时显存，将计算任务提交到指定的 **Stream** 中异步执行。

# 句柄 (Handle)

CUDA 数学库的上下文管理器，比如 `cublasHandle_t`、`cudnnHandle_t`、`cusparseHandle_t` 。

内部维护库运行所需的硬件状态、绑定的 Stream、默认计算模式等资源。

调用库函数时通常必须传入句柄，它相当于库在当前 GPU 设备上的会话实例。

# 矩阵乘法操作描述符 (Matmul Operation Descriptor)

用于定义算什么以及计算规则，比如 `cublasLtMatmulDesc_t` 、`cudnnConvolutionDescriptor_t` 。

负责指定计算精度 (如 FP16、TF32) 、缩放因子类型、矩阵是否转置，以及 Epilogue 算子融合 (如矩阵乘法后接上 Bias 或 ReLU/GELU) 。

# 矩阵布局描述符 (Matrix Layout Descriptor)

用于定义数据在显存中长什么样，比如 `cublasLtMatrixLayout_t`、`cudnnTensorDescriptor_t` 。

记录了数据类型、维度大小、主序 (行主序/列主序) 、Leading Dimension (跨度/步长) 以及 Batch 偏移量。

# 偏好描述符 (Preference Descriptor)

告诉库的启发式搜索 (Heuristic) 引擎怎么选算法。比如 `cublasLtMatmulPreference_t` 。

通过它可以设置约束条件，最常见的是限制最大可用 WorkSpace 大小、指定对齐要求或归约策略，帮助库在性能与显存占用之间筛选出最合适的底层 Kernel。

# WorkSpace (工作区 / 临时显存)

由用户预先在 GPU 显存中分配、专门提供给库函数用于存放中间计算结果。

广泛存在于 cuBLAS、cuDNN、cuSOLVER、cuSPARSE 等库中。

让用户自己管理 WorkSpace 是为了避免库内部频繁调用 `cudaMalloc` 和 `cudaFree` ，否则导致严重的全局同步和性能开销，同时也能解锁需要额外显存的高性能算法，如 Split-K 。

# Stream (流)

CUDA Runtime/Driver 层的核心异步执行机制 (`cudaStream_t`) ，本质上是 GPU 上的一个先进先出任务队列。

同一个 Stream 内的 Kernel 启动和显存拷贝会严格按顺序执行，不同 Stream 之间的任务则可以并发或重叠执行，例如在计算的同时进行 `cudaMemcpyAsync` 数据传输。

各数学库既可以通过句柄绑定 Stream，如 `cudnnSetStream` ，也可以在执行接口中直接传入。