//
// Created by king on 1/5/17.
//

#ifndef CAFFE_MEM_POOL_H
#define CAFFE_MEM_POOL_H

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    class MemPool {
    public:
        inline static MemPool &Get() {
            if (!singleton_.get()) {
                singleton_.reset(new MemPool());
            }
            return *singleton_;
        }

        static size_t size() {return Get().size_;}
        static void* device_ptr() {return Get().device_ptr_;}
        static void* host_ptr() {return Get().host_ptr_;}

        static void add(size_t size);

        static void make();

        static void * alloc_device(size_t size);
        static void * alloc_host(size_t size);

        static void destroy();

    private:
        MemPool() : device_ptr_(NULL), host_ptr_(NULL), 
               size_(0), device_offset_(0), host_offset_(0), 
               inited_(false) {}

    protected:
        void *device_ptr_;
        void *host_ptr_;
        size_t size_;
        size_t device_offset_;
        size_t host_offset_;
        bool inited_;

        static shared_ptr<MemPool> singleton_;
    };
}


#endif //CAFFE_MEM_POOL_H
