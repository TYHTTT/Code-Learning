#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

constexpr int M = 3;
constexpr int K = 4;
constexpr int N = 2;

// CUDA 错误检查：打印错误并返回状态码。
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess) {
        printf("CUDA error: code=%d, name=%s, description=%s, file=%s, line=%d\n",
               static_cast<int>(error_code),
               cudaGetErrorName(error_code),
               cudaGetErrorString(error_code),
               filename,
               lineNumber);
    }

    return error_code;
}

// cuBLAS 错误检查：打印错误并返回状态码。
cublasStatus_t CublasErrorCheck(cublasStatus_t error_code, const char* filename, int lineNumber)
{
    if (error_code != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: code=%d, file=%s, line=%d\n",
               static_cast<int>(error_code),
               filename,
               lineNumber);
    }

    return error_code;
}

// 按行主序打印矩阵。
void PrintMatrix(const float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU 矩阵乘法。
void CpuMatmul(const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    const float A[M * K] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    const float B[K * N] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float C_cpu[M * N];
    float C_cublas_s[M * N];
    float C_cublas_h[M * N];

    // CPU 计算。
    CpuMatmul(A, B, C_cpu);

    // 创建 cuBLAS 句柄。
    cublasHandle_t handle;
    CublasErrorCheck(cublasCreate(&handle), __FILE__, __LINE__);

    // 分配单精度显存。
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    ErrorCheck(cudaMalloc(&d_A, M * K * sizeof(float)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc(&d_B, K * N * sizeof(float)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc(&d_C, M * N * sizeof(float)), __FILE__, __LINE__);

    // 将单精度输入复制到显存。
    ErrorCheck(
        cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice),
        __FILE__, __LINE__
    );

    ErrorCheck(
        cudaMemcpy(d_B, B, sizeof(B), cudaMemcpyHostToDevice),
        __FILE__, __LINE__
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 单精度矩阵乘法。
    // cuBLAS 按列主序解释数据，利用 Cᵀ = Bᵀ × Aᵀ。
    CublasErrorCheck(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N
        ),
        __FILE__, __LINE__
    );

    // 单精度结果复制回 CPU。
    ErrorCheck(
        cudaMemcpy(
            C_cublas_s,
            d_C,
            sizeof(C_cublas_s),
            cudaMemcpyDeviceToHost
        ),
        __FILE__, __LINE__
    );

    // 在 CPU 上将输入转换为 half。
    half A_h[M * K];
    half B_h[K * N];

    for (int i = 0; i < M * K; ++i) {
        A_h[i] = __float2half(A[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        B_h[i] = __float2half(B[i]);
    }

    // 分配半精度显存。
    half* d_A_h = nullptr;
    half* d_B_h = nullptr;
    half* d_C_h = nullptr;

    ErrorCheck(cudaMalloc(&d_A_h, M * K * sizeof(half)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc(&d_B_h, K * N * sizeof(half)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc(&d_C_h, M * N * sizeof(half)), __FILE__, __LINE__);

    // 将半精度输入复制到显存。
    ErrorCheck(
        cudaMemcpy(d_A_h, A_h, sizeof(A_h), cudaMemcpyHostToDevice),
        __FILE__, __LINE__
    );

    ErrorCheck(
        cudaMemcpy(d_B_h, B_h, sizeof(B_h), cudaMemcpyHostToDevice),
        __FILE__, __LINE__
    );

    const half alpha_h = __float2half(1.0f);
    const half beta_h = __float2half(0.0f);

    // 半精度矩阵乘法。
    CublasErrorCheck(
        cublasHgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            d_B_h, N,
            d_A_h, K,
            &beta_h,
            d_C_h, N
        ),
        __FILE__, __LINE__
    );

    // 半精度结果复制回 CPU。
    half C_h[M * N];

    ErrorCheck(
        cudaMemcpy(
            C_h,
            d_C_h,
            sizeof(C_h),
            cudaMemcpyDeviceToHost
        ),
        __FILE__, __LINE__
    );

    // 转换成 float，便于统一打印。
    for (int i = 0; i < M * N; ++i) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // 打印输入和三种计算结果。
    printf("Matrix A (%dx%d):\n", M, K);
    PrintMatrix(A, M, K);

    printf("Matrix B (%dx%d):\n", K, N);
    PrintMatrix(B, K, N);

    printf("CPU Result (%dx%d):\n", M, N);
    PrintMatrix(C_cpu, M, N);

    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PrintMatrix(C_cublas_s, M, N);

    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PrintMatrix(C_cublas_h, M, N);

    // 释放显存。
    ErrorCheck(cudaFree(d_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_C), __FILE__, __LINE__);

    ErrorCheck(cudaFree(d_A_h), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_B_h), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_C_h), __FILE__, __LINE__);

    // 销毁 cuBLAS 句柄。
    CublasErrorCheck(cublasDestroy(handle), __FILE__, __LINE__);

    return 0;
}