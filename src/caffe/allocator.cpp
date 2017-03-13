#include "caffe/allocator.hpp"
#include "caffe/mem_pool.hpp"

namespace caffe {

// Theoretically, CaffeMallocHost and CaffeFreeHost should simply call the
// cudaMallocHost and cudaFree functions in order to create pinned memory.
// However, those codes rely on the existence of a cuda GPU (I don't know
// why that is a must since allocating memory should not be accessing the
// GPU resource, but it just creates an error as of Cuda 5.0) and will cause
// problem when running on a machine without GPU. Thus, we simply define
// these two functions for safety and possible future change if the problem
// of calling cuda functions disappears in a future version.
//
// In practice, although we are creating unpinned memory here, as long as we
// are constantly accessing them the memory pages almost always stays in
// the physical memory (assuming we have large enough memory installed), and
// does not seem to create a memory bottleneck here.
void CaffeMallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

void CaffeFreeHost(void* ptr) {
  free(ptr);
}

/* ------------------- DefaultAllocator ----------------- */
void * DefaultAllocator::malloc_device(int size, bool* own_data) {
  void * ptr = NULL;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  *own_data = true;
  return ptr;
}

void * DefaultAllocator::malloc_host(int size, bool * own_data) {
  void * ptr = NULL;
  CaffeMallocHost(&ptr, size);
  *own_data = true;
  return ptr;
}

void DefaultAllocator::mfree_device(void * ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

void DefaultAllocator::mfree_host(void * ptr) {
  CaffeFreeHost(ptr);
}

/* ------------------- PoolAllocator ----------------- */

void * PoolAllocator::malloc_device(int size, bool* own_data) {
  *own_data = false;
  return MemPool::Get().alloc_device(size);
}

void * PoolAllocator::malloc_host(int size, bool* own_data) {
  *own_data = false;
  return MemPool::Get().alloc_host(size);
}

void PoolAllocator::mfree_device(void * ptr) {
  return;
}

void PoolAllocator::mfree_host(void * ptr) {
  return;
}


/************************* Unify Allocator ****************************/

void* UnifyAllocator::malloc_device(int size, bool* own_data) {
  void * ptr = NULL;
  CUDA_CHECK(cudaMallocManaged(&ptr, size));
  *own_data = true;
  return ptr;
}

void* UnifyAllocator::malloc_host(int size, bool * own_data) {
  void * ptr = NULL;
  CaffeMallocHost(&ptr, size);
  *own_data = true;
  return ptr;
}

void UnifyAllocator::mfree_device(void * ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

void UnifyAllocator::mfree_host(void * ptr) {
  CaffeFreeHost(ptr);
}

}
