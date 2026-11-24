#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasXt.h>

// CUDA 错误检查：打印错误并返回状态码。
cudaError_t cudaErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
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
cublasStatus_t cublasErrorCheck(cublasStatus_t error_code, const char* filename, int lineNumber)
{
    if (error_code != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: code=%d, name=%s, descriptions=%s, file=%s, line=%d\n",
               static_cast<int>(error_code),
               cublasGetStatusName(error_code),
               cublasGetStatusString(error_code),
               filename,
               lineNumber);
    }

    return error_code;
}

void PrintMatrix(const float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%10.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(){

    //侦测可用GPU
    int deviceCount = 0;
    cudaErrorCheck(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__);
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return -1;
    }

    //可用GPU列表
    int* deviceList = (int*)malloc(deviceCount * sizeof(int));
    if (deviceList == nullptr) {
        printf("Failed to allocate memory for device list.\n");
        return -1;
    }
    
    // 初始化设备列表
    for(int i = 0; i < deviceCount; ++i) {
        deviceList[i] = i;
    }

    // cuBLASXt 通常面向大规模矩阵乘法，适用于多 GPU 系统。它可以在多个 GPU 上分配计算任务，从而提高性能。对于小规模矩阵乘法，使用 cuBLAS 库可能更合适，因为它在单个 GPU 上的性能已经非常高。
    const int M = 2048;
    const int K = 1024;
    const int N = 512;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    //使用 malloc 在堆 (Heap) 上分配主机内存，单个 2048x2048 的 float 矩阵就占用约 16 MB，如果直接在函数内定义局部数组会直接导致栈溢出崩溃；
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // 检查内存分配是否成功
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        printf("Failed to allocate host memory.\n");
        free(h_A);
        free(h_B);
        free(h_C);
        free(deviceList);
        return -1;
    }

    //初始化矩阵 A 、B、C
    for (int i = 0;i<M*K;++i) h_A[i] = 1.0f; // A 矩阵初始化为全 1
    for (int i = 0;i<K*N;++i) h_B[i] = 2.0f; // B 矩阵初始化为全 2
    for (int i = 0;i<M*N;++i) h_C[i] = 0.0f; // C 矩阵初始化为全 0

    //创建 cuBLASXt 句柄
    cublasXtHandle_t handle;
    cublasErrorCheck(cublasXtCreate(&handle), __FILE__, __LINE__);
    cublasErrorCheck(cublasXtDeviceSelect(handle, deviceCount, deviceList), __FILE__, __LINE__);//把电脑里所有检测到的 NVIDIA 显卡都用上

    //执行矩阵乘法 C = alpha * A * B + beta * C
    //因为 cuBLASXt 默认列主序，利用 (A * B)^T = B^T * A^T，
    cublasErrorCheck(cublasXtSgemm(handle, 
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        h_B, N, //B有N列
        h_A, K, //A有K列
        &beta, 
        h_C, N),
        __FILE__, __LINE__);

    // 验证结果
    float expected = 2.0f * K;
    int correct = 1;
    // 只检查前 10 个元素以验证结果
    for (int i = 0; i < 10; ++i) {
        if (fabsf(h_C[i] - expected) > 1e-3f) {
            fprintf(stderr, "verification failed! index %d: actual=%f, expected=%f\n", i, h_C[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Computation Finshed! h_C[0] = %.1f (expected: %.1f)\n", h_C[0], expected);
    }

    //释放资源
    cublasErrorCheck(cublasXtDestroy(handle), __FILE__, __LINE__);
    free(h_A);
    free(h_B);
    free(h_C);
    free(deviceList);

    return 0;

}