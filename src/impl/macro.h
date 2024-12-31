#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkKernelErrors(expr)                                                         \
  do {                                                                                  \
    expr;                                                                               \
                                                                                        \
    cudaError_t __err = cudaGetLastError();                                             \
    if (__err != cudaSuccess) {                                                         \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
      abort();                                                                          \
    }                                                                                   \
  } while (0)

// CUDA Runtime error messages
static const char* _cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

// cuBLAS API errors
static const char* _cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

