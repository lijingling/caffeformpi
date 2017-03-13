#ifndef CAFFE_ALLOCTOR_HPP_
#define CAFFE_ALLOCTOR_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

void CaffeMallocHost(void** ptr, size_t size);
void CaffeFreeHost(void* ptr);

class Allocator {
  public:
    virtual void * malloc_device(int size, bool* own_data) = 0;
    virtual void * malloc_host(int size, bool* own_data) = 0;

    virtual void mfree_device(void * ptr) = 0;
    virtual void mfree_host(void * ptr) = 0;
};

class DefaultAllocator: public Allocator {
  public:
    void * malloc_device(int size, bool* own_data);
    void * malloc_host(int size, bool* own_data);

    void mfree_device(void * ptr);
    void mfree_host(void * ptr);
};

class PoolAllocator: public Allocator {
  public:
    void * malloc_device(int size, bool* own_data);
    void * malloc_host(int size, bool* own_data);

    void mfree_device(void * ptr);
    void mfree_host(void * ptr);
};

class UnifyAllocator: public Allocator{
  public:
    void * malloc_device(int size, bool* own_data);
    void * malloc_host(int size, bool* own_data);

    void mfree_device(void * ptr);
    void mfree_host(void * ptr);
};

}
#endif
