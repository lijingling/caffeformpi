#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_dgmm<float>(const CBLAS_SIDE SideX, int m, int n, 
  const float* A, const float* X, float* y) {
  cublasSideMode_t mode = 
    (SideX == CblasLeft) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  CUBLAS_CHECK(cublasSdgmm(Caffe::cublas_handle(), mode, m, n, A, m, X, 1, y, m));
}

template <>
void caffe_gpu_dgmm<double>(const CBLAS_SIDE SideX, int m, int n, 
  const double* A, const double* X, double* y) {
  cublasSideMode_t mode = 
    (SideX == CblasLeft) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  CUBLAS_CHECK(cublasDdgmm(Caffe::cublas_handle(), mode, m, n, A, m, X, 1, y, m));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

/* ----------------- caffe_gpu_rmax  -------------------- */

template <typename Dtype>
__global__ void max_idx_kernel_reduction_within_block(
    const Dtype *data, Dtype * blk_vals, 
    const int xSize, const int ySize, const int max_block_x){
  __shared__ volatile Dtype vals[COMPATIBLE_CUDA_NUM_THREADS]; 

  int idx = threadIdx.x+blockDim.x * blockIdx.x; 
  int idy = blockIdx.y;
  Dtype my_val = -FLT_MAX; 
  while (idx < xSize){   
    Dtype temp = data[idy*xSize+idx];
    if (temp > my_val) {
      my_val = temp;
    }
    idx += blockDim.x*gridDim.x;
  }                                       
  vals[threadIdx.x] = my_val;
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1){
    if (threadIdx.x < i) {
      if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
        vals[threadIdx.x] = vals[threadIdx.x+i]; 
      }             
    }      
    __syncthreads();
  }     
    
  if (!threadIdx.x){   
    blk_vals[blockIdx.y * max_block_x + blockIdx.x] = vals[0];         
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void max_idx_kernel_final(const Dtype * blk_vals, const int max_block_x, 
    Dtype *result_maxVal) {
  __shared__ volatile Dtype vals[COMPATIBLE_CUDA_NUM_THREADS]; 

  int idx = threadIdx.x;
  int idy = blockIdx.y;
  Dtype my_val = -FLT_MAX;
  while (idx < max_block_x ){
    Dtype temp = blk_vals[idy * max_block_x + idx];
    if (temp > my_val) {
      my_val = temp;
    }
    idx += blockDim.x;
  } 
    
  idx = threadIdx.x;
  vals[idx] = my_val;  
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1) {
    if (idx < i) {
      if (vals[idx] < vals[idx + i]) {
        vals[idx] = vals[idx+i]; 
      }
    }
    __syncthreads();
  }
  if(!threadIdx.x){
    result_maxVal[idy] = vals[0];
  }
}

template <typename Dtype>
void caffe_gpu_rmax(const int ncol, const int nrow,
    const Dtype* x, Dtype* y, Dtype* blk_vals) {
  const int max_block_x = (ncol / COMPATIBLE_CUDA_NUM_THREADS) + 1;
  dim3 grids(max_block_x, nrow);
  dim3 threads(COMPATIBLE_CUDA_NUM_THREADS,1);
  dim3 grids2(1,nrow);
  dim3 threads2(COMPATIBLE_CUDA_NUM_THREADS);
  // NOLINT_NEXT_LINE(whitespace/operators)
  max_idx_kernel_reduction_within_block<Dtype><<<grids, threads>>>(x, blk_vals,
    ncol, nrow, max_block_x);
  // NOLINT_NEXT_LINE(whitespace/operators)
  max_idx_kernel_final<Dtype><<<grids2,threads2>>>(blk_vals, max_block_x, y);
}

template void caffe_gpu_rmax(const int ncol, const int nrow,
    const float* x, float* y, float* blk_vals);
template void caffe_gpu_rmax(const int ncol, const int nrow,
    const double* x, double* y, double* blk_vals);

/* ----------------- caffe_gpu_rmin  -------------------- */

template <typename Dtype>
__global__ void min_idx_kernel_reduction_within_block(
    const Dtype *data, Dtype * blk_vals, Dtype * blk_idxs, 
    const int xSize, const int ySize, const int max_block_x){
  __shared__ volatile float vals[COMPATIBLE_CUDA_NUM_THREADS];
  __shared__ volatile Dtype idxs[COMPATIBLE_CUDA_NUM_THREADS]; 
  int idx = threadIdx.x+blockDim.x * blockIdx.x; 
  int idy = blockIdx.y;
  Dtype my_val = FLT_MAX; 
  Dtype my_idx = Dtype(-1);
  while (idx < xSize){   
    Dtype temp = data[idy*xSize+idx];
    if (temp < my_val) {
      my_val = temp;
      my_idx = Dtype(idx);
    }
    idx += blockDim.x*gridDim.x;
  }                                       
  vals[threadIdx.x] = my_val;
  idxs[threadIdx.x] = my_idx;
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1){
    if (threadIdx.x < i) {
      if (vals[threadIdx.x] > vals[threadIdx.x + i]) {
        vals[threadIdx.x] = vals[threadIdx.x+i]; 
        idxs[threadIdx.x] = idxs[threadIdx.x+i];
      }             
    }      
    __syncthreads();
  }     
    
  if (!threadIdx.x){   
    blk_vals[blockIdx.y * max_block_x + blockIdx.x] = vals[0];
    blk_idxs[blockIdx.y * max_block_x + blockIdx.x] = idxs[0];         
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void min_idx_kernel_final(const Dtype * blk_vals, const Dtype * blk_idxs,
    const int max_block_x, Dtype *result_maxVal, Dtype *result_maxIdx) {
  __shared__ volatile Dtype vals[COMPATIBLE_CUDA_NUM_THREADS]; 
  __shared__ volatile Dtype idxs[COMPATIBLE_CUDA_NUM_THREADS]; 
  int idx = threadIdx.x;
  int idy = blockIdx.y;
  Dtype my_val = FLT_MAX;
  Dtype my_idx = Dtype(-1);
  while (idx < max_block_x ){
    Dtype temp = blk_vals[idy * max_block_x + idx];
    if (temp < my_val) {
      my_val = temp;
      my_idx = blk_idxs[idy * max_block_x + idx];
    }
    idx += blockDim.x;
  } 
    
  idx = threadIdx.x;
  vals[idx] = my_val; 
  idxs[idx] = my_idx;
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1) {
    if (idx < i) {
      if (vals[idx] > vals[idx + i]) {
        vals[idx] = vals[idx+i]; 
        idxs[idx] = idxs[idx+i];
      }
    }
    __syncthreads();
  }
  if(!threadIdx.x){
    result_maxVal[idy] = vals[0];
    result_maxIdx[idy] = idxs[0];
  }
}

template <typename Dtype>
void caffe_gpu_rmin(const int ncol, const int nrow,
    const Dtype* x, Dtype* y, Dtype* y_idx, Dtype* blk_vals, Dtype* blk_idxs) {
  const int max_block_x = (ncol / COMPATIBLE_CUDA_NUM_THREADS) + 1;
  dim3 grids(max_block_x, nrow);
  dim3 threads(COMPATIBLE_CUDA_NUM_THREADS,1);
  dim3 grids2(1,nrow);
  dim3 threads2(COMPATIBLE_CUDA_NUM_THREADS);
  min_idx_kernel_reduction_within_block<Dtype><<<grids, threads>>>(x, blk_vals, blk_idxs,
    ncol, nrow, max_block_x);
  min_idx_kernel_final<Dtype><<<grids2,threads2>>>(blk_vals, blk_idxs, max_block_x, y, y_idx);
}

template void caffe_gpu_rmin(const int ncol, const int nrow,
    const float* x, float* y, float* y_idx, float* blk_vals, float* blk_idxs);
template void caffe_gpu_rmin(const int ncol, const int nrow,
    const double* x, double* y, double* y_idx, double* blk_vals, double* blk_idxs);

/* ----------------- caffe_gpu_rsum  -------------------- */

template <typename Dtype>
__global__ void sum_kernel_reduction_within_block(
    const Dtype *data, Dtype * blk_vals, 
    const int xSize, const int ySize, const int max_block_x){
  __shared__ volatile Dtype vals[COMPATIBLE_CUDA_NUM_THREADS]; 

  int idx = threadIdx.x+blockDim.x * blockIdx.x; 
  int idy = blockIdx.y;
  Dtype my_val = 0; 
  while (idx < xSize){   
    my_val += data[idy*xSize+idx];
    idx += blockDim.x*gridDim.x;
  }                                       
  vals[threadIdx.x] = my_val;
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1){
    if (threadIdx.x < i) {
      vals[threadIdx.x] += vals[threadIdx.x+i];   
    }      
    __syncthreads();
  }     
    
  if (!threadIdx.x){   
    blk_vals[blockIdx.y * max_block_x + blockIdx.x] = vals[0];         
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void sum_kernel_final(const Dtype * blk_vals, const int max_block_x, 
    Dtype *result_sumVal) {
  __shared__ volatile Dtype vals[COMPATIBLE_CUDA_NUM_THREADS]; 

  int idx = threadIdx.x;
  int idy = blockIdx.y;
  Dtype my_val = 0;
  while (idx < max_block_x ){
    my_val = blk_vals[idy * max_block_x + idx]; 
    idx += blockDim.x;
  } 
    
  idx = threadIdx.x;
  vals[idx] = my_val;  
  __syncthreads();
  for (int i = (COMPATIBLE_CUDA_NUM_THREADS>>1); i > 0; i>>=1) {
    if (idx < i) {
      vals[idx] += vals[idx+i]; 
    }
    __syncthreads();
  }
  if(!threadIdx.x){
    result_sumVal[idy] = vals[0];
  }
}

template <typename Dtype>
void caffe_gpu_rsum(const int ncol, const int nrow,
    const Dtype* x, Dtype* y, Dtype* blk_vals) {
  const int max_block_x = (ncol / COMPATIBLE_CUDA_NUM_THREADS) + 1;
  dim3 grids(max_block_x, nrow);
  dim3 threads(COMPATIBLE_CUDA_NUM_THREADS,1);
  dim3 grids2(1,nrow);
  dim3 threads2(COMPATIBLE_CUDA_NUM_THREADS);
  // NOLINT_NEXT_LINE(whitespace/operators)
  sum_kernel_reduction_within_block<Dtype><<<grids, threads>>>(x, blk_vals,
    ncol, nrow, max_block_x);
  // NOLINT_NEXT_LINE(whitespace/operators)
  sum_kernel_final<Dtype><<<grids2,threads2>>>(blk_vals, max_block_x, y);
}

template void caffe_gpu_rsum(const int ncol, const int nrow,
    const float* x, float* y, float* blk_vals);
template void caffe_gpu_rsum(const int ncol, const int nrow,
    const double* x, double* y, double* blk_vals);

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

__global__ void popc_kernel(const int n, const float* a,
    const float* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(a[index]) ^
                      static_cast<uint32_t>(b[index]));
  }
}

__global__ void popcll_kernel(const int n, const double* a,
    const double* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
                      static_cast<uint64_t>(b[index]));
  }
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        /* NOLINT_NEXT_LINE(build/include_what_you_use) */
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
