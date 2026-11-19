#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

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
    const int M = 4;
    const int K = 3;
    const int N = 2;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float h_A_fp32[M*K] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };

    const float h_B_fp32[K*N] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    // 将 FP32 数据转换为 FP16
    __half h_A_fp16[M*K];
    __half h_B_fp16[K*N];
    for (int i = 0; i < M*K; ++i) {
        h_A_fp16[i] = __float2half(h_A_fp32[i]);
    }
    for (int i = 0; i < K*N; ++i) {
        h_B_fp16[i] = __float2half(h_B_fp32[i]);
    }

    __half h_C_fp16[M*N];//接收GPU计算结果

    __half* d_A_fp16;
    __half* d_B_fp16;
    __half* d_C_fp16;

    cudaErrorCheck(cudaMalloc(&d_A_fp16, M * K * sizeof(__half)), __FILE__, __LINE__);
    cudaErrorCheck(cudaMalloc(&d_B_fp16, K * N * sizeof(__half)), __FILE__, __LINE__);
    cudaErrorCheck(cudaMalloc(&d_C_fp16, M * N * sizeof(__half)), __FILE__, __LINE__);

    cudaErrorCheck(cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(__half), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cudaErrorCheck(cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(__half), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // 创建 cuBLASLt 句柄
    cublasLtHandle_t cublasLtHandle;
    cublasErrorCheck(cublasLtCreate(&cublasLtHandle), __FILE__, __LINE__);

    // 创建 CUDA Stream
    cudaStream_t stream;
    cudaErrorCheck(cudaStreamCreate(&stream), __FILE__, __LINE__);
    
    //创建矩阵乘法操作描述符 (Matmul Descriptor)
    cublasLtMatmulDesc_t opreationDesc=nullptr;
    cublasErrorCheck(cublasLtMatmulDescCreate(&opreationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F), __FILE__, __LINE__);//保持 CUBLAS_COMPUTE_32F 和 CUDA_R_32F，实现“FP16 乘法，FP32 累加”
    cublasOperation_t transA = CUBLAS_OP_N; // A 不转置
    cublasOperation_t transB = CUBLAS_OP_N; // B 不转置
    cublasErrorCheck(cublasLtMatmulDescSetAttribute(opreationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatmulDescSetAttribute(opreationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)), __FILE__, __LINE__);

    //创建行主序矩阵布局描述符 (Matrix Layout Descriptor)，数据类型改为 CUDA_R_16F
    cublasLtMatrixLayout_t Adesc=nullptr;
    cublasLtMatrixLayout_t Bdesc=nullptr;
    cublasLtMatrixLayout_t Cdesc=nullptr;
    cublasErrorCheck(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, K), __FILE__, __LINE__); // A 的行主序布局描述符
    cublasErrorCheck(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, N), __FILE__, __LINE__); // B 的行主序布局描述符
    cublasErrorCheck(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, N), __FILE__, __LINE__); // C 的行主序布局描述符
    
    // 显式指定存储顺序为行主序 (CUBLASLT_ORDER_ROW)
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW; // 行主序
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), __FILE__, __LINE__);

    // 创建偏好描述符与 WorkSpace
    cublasLtMatmulPreference_t preference=nullptr;
    cublasErrorCheck(cublasLtMatmulPreferenceCreate(&preference), __FILE__, __LINE__);
    size_t workspaceSize = 4 * 1024 * 1024; // 4MB
    void* d_workspace = nullptr;
    cudaErrorCheck(cudaMalloc(&d_workspace, workspaceSize), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)), __FILE__, __LINE__);

    // 查询启发式最优算法
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasErrorCheck(cublasLtMatmulAlgoGetHeuristic(cublasLtHandle, opreationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults), __FILE__, __LINE__);
    if (returnedResults == 0) {
        printf("No suitable cuBLASLt algorithm found.\n");
        return -1;
    }

    // 执行矩阵乘法
    cublasErrorCheck(
        cublasLtMatmul(
            cublasLtHandle, 
            opreationDesc, 
            &alpha, 
            d_A_fp16, Adesc, 
            d_B_fp16, Bdesc, 
            &beta, 
            d_C_fp16, Cdesc, 
            d_C_fp16, Cdesc, 
            &heuristicResult.algo, 
            d_workspace, 
            workspaceSize, 
            stream
        ), 
        __FILE__, __LINE__
    );

    // 同步并拷贝回至内存
    cudaErrorCheck(cudaStreamSynchronize(stream), __FILE__, __LINE__);
    cudaErrorCheck(cudaMemcpy(h_C_fp16, d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // 结果转换回 FP32 并打印
    float h_C_fp32[M*N];
    for (int i = 0; i < M*N; ++i) {
        h_C_fp32[i] = __half2float(h_C_fp16[i]);
    }
    PrintMatrix(h_C_fp32, M, N);

    //资源释放
    cublasErrorCheck(cublasLtMatmulPreferenceDestroy(preference), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatrixLayoutDestroy(Adesc), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatrixLayoutDestroy(Bdesc), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatrixLayoutDestroy(Cdesc), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtMatmulDescDestroy(opreationDesc), __FILE__, __LINE__);
    cublasErrorCheck(cublasLtDestroy(cublasLtHandle), __FILE__, __LINE__);
    cudaErrorCheck(cudaStreamDestroy(stream), __FILE__, __LINE__);
    cudaErrorCheck(cudaFree(d_workspace), __FILE__, __LINE__);
    cudaErrorCheck(cudaFree(d_A_fp16), __FILE__, __LINE__);
    cudaErrorCheck(cudaFree(d_B_fp16), __FILE__, __LINE__);
    cudaErrorCheck(cudaFree(d_C_fp16), __FILE__, __LINE__);

    return 0;

}