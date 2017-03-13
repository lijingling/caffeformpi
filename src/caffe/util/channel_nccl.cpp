#ifdef USE_MPI

#include "caffe/util/channel_nccl.hpp"

#include "caffe/common.hpp"

#include "nccl.h"

namespace caffe {

shared_ptr<NcclComm> NcclComm::singleton_;

void NcclComm::DestroyAll() {
 for(int i = 0; i < streams_.size(); ++i) {
   CUDA_CHECK(cudaStreamDestroy(streams_[i]));
 }
}

void NcclComm::WaitAll() {
 for(int i = 0; i < streams_.size(); ++i) {
   CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
 }
 n_runing_job_ = 0;
}

void NcclComm::AddJob(MPIJob & job) {
  ncclDataType_t data_type = (job.dtype_size_ == 4) ? ncclFloat : ncclDouble;
  bool reuse_stream = true;
  if(n_runing_job_ + 1 > streams_.size()) {
   reuse_stream = false;
   streams_.resize(streams_.size() + 1);
  }
  cudaStream_t& stream = streams_[n_runing_job_];
  if(!reuse_stream) {
   CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // call Nccl APIs for real works
  CUDA_CHECK(cudaDeviceSynchronize());
  switch (job.op_) {
    case OP_SUM_ALL: {
      NCCL_CHECK(ncclAllReduce(job.src_ptr_, job.dst_ptr_, job.count_, data_type,
                              ncclSum, Caffe::NCCL_COMM(), stream));
      break;
    }
    case OP_MAX_ALL: {
      NCCL_CHECK(ncclAllReduce(job.src_ptr_, job.dst_ptr_, job.count_, data_type,
                              ncclMax, Caffe::NCCL_COMM(), stream));
      break;
    }
    case OP_GATHER_ALL: {
      NCCL_CHECK(ncclAllGather(job.src_ptr_, job.count_, data_type, job.dst_ptr_,
                              Caffe::NCCL_COMM(), stream));
      break;
    }
    case OP_SCATTER: {
      int rank = Caffe::MPI_my_rank();
      int nbyte = job.dtype_size_ * job.count_;
      CUDA_CHECK(cudaMemcpyAsync(job.dst_ptr_, 
          (const void *)((const uint8_t*)job.src_ptr_ + nbyte * rank), 
          nbyte, 
          cudaMemcpyDeviceToDevice, stream));
      break;
    }
    case OP_BROADCAST: {
      NCCL_CHECK(ncclBcast(job.src_ptr_, job.count_, data_type, 0,
                              Caffe::NCCL_COMM(), stream));
      break;
    }
    default: {
      LOG(FATAL)<<"Unknown NCCL job type";
    }
  }
  n_runing_job_++;
}

}

#endif //USE_MPI
