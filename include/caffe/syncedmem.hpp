#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/allocator.hpp"

namespace caffe {

class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false), gpu_device_(-1) {
    allocator_.reset(new DefaultAllocator());
  }
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false), gpu_device_(-1) {
    allocator_.reset(new DefaultAllocator());
  }
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

  void Resize(size_t new_size);

  void set_allocator_type(allocator_type_t t) {
    switch(t) {
      case EDefaultAllocator:
        allocator_.reset(new DefaultAllocator());
        break;
      case EPoolAllocator:
        allocator_.reset(new PoolAllocator());
        break;
      case EUnifyAllocator:
        allocator_.reset(new UnifyAllocator());
	break;
      default:
        LOG(FATAL) << "unrecognized allocator type";
        break;
    }
  }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool own_gpu_data_;
  int gpu_device_;
  shared_ptr<Allocator> allocator_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
