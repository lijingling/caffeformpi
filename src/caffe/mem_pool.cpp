//
// Created by king on 1/5/17.
//

#include "caffe/mem_pool.hpp"
#include "caffe/allocator.hpp"

namespace caffe{

    shared_ptr<MemPool> MemPool::singleton_;
    
    void MemPool::add(size_t size){
        CHECK(!(Get().inited_)) << "no extend after initializing.";
        Get().size_ += size;
    }

    void MemPool::make() {
        if(Get().size_ <= 0) {
          return;
        }
        CHECK(!(Get().inited_)) << "mem pool has already been initialized.";
        CUDA_CHECK(cudaMalloc(&Get().device_ptr_, Get().size_));
        CUDA_CHECK(cudaMemset(Get().device_ptr_, 0, Get().size_));
        CaffeMallocHost(&Get().host_ptr_, Get().size_);
        CHECK(Get().host_ptr_);
        caffe_memset(Get().size_, 0, Get().host_ptr_);
        Get().inited_ = true;
        LOG(INFO) << "make memory pool with size = " << 
            Get().size_ / 1024.0 / 1024 << "MB";
    }

    void MemPool::destroy() {
      if(Get().inited_ && Get().size_ > 0) {
        if(Get().device_ptr_) {
          CUDA_CHECK(cudaFree(Get().device_ptr_));
          Get().device_ptr_ = NULL;
        }
        if(Get().host_ptr_) {
          CaffeFreeHost(Get().host_ptr_);
          Get().host_ptr_ = NULL;
        }
      }
    }

    void* MemPool::alloc_device(size_t size){
        CHECK(Get().inited_) << "mem pool has not been initialized.";
        CHECK_LE(Get().device_offset_ + size, Get().size_) 
            << "runing out of memory in pool";
        char* ret = reinterpret_cast<char*>(Get().device_ptr_) + Get().device_offset_;
        Get().device_offset_ += size;

        return reinterpret_cast<void*>(ret);
    }

    void* MemPool::alloc_host(size_t size){
        CHECK(Get().inited_) << "mem pool has not been initialized.";
        CHECK_LE(Get().host_offset_ + size, Get().size_) 
            << "runing out of memory in pool";
        char* ret = reinterpret_cast<char*>(Get().host_ptr_) + Get().host_offset_;
        Get().host_offset_ += size;

        return reinterpret_cast<void*>(ret);
    }
}
