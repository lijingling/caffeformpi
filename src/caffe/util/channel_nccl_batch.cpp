#ifdef USE_MPI

#include "caffe/util/channel_nccl_batch.hpp"

#include "caffe/common.hpp"

#include "nccl.h"

namespace caffe {

shared_ptr<NcclBatchComm> NcclBatchComm::singleton_;

void NcclBatchComm::DestroyAll() {
 for(int i = 0; i < streams_.size(); ++i) {
   CUDA_CHECK(cudaStreamDestroy(streams_[i]));
 }
}

void NcclBatchComm::WaitAll() {
 for(int i = 0; i < streams_.size(); ++i) {
   CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
 }
 n_runing_job_ = 0;
}

void NcclBatchComm::AddJob(MPIJob & job, bool force_to_go) {
  comm_count_ += job.count_;
  float comm_size = comm_count_ * job.dtype_size_ / 1024.0 / 1024; 
  if(!force_to_go && 
      comm_mode_ == SolverParameter_CommMode_BATCH_WISE && comm_size < comm_batch_size_) {
    head_job_ = job;
    head_job_.count_ = 0;
    return;
  }

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
      NCCL_CHECK(ncclAllReduce(job.src_ptr_, job.dst_ptr_, comm_count_, data_type,
                              ncclSum, Caffe::NCCL_COMM(), stream));
      break;
    }
    default: {
      LOG(FATAL)<<"Unknown NCCL job type";
    }
  }
  head_job_.src_ptr_ = NULL;
  head_job_.dst_ptr_ = NULL;
  head_job_.count_ = 0;
  comm_count_ = 0;
  n_runing_job_++;
}

void NcclBatchComm::FlushJob() {
  if(head_job_.src_ptr_ && head_job_.dst_ptr_ && 
     comm_count_ > 0) {
    AddJob(head_job_, true);
  }
}

}

#endif //USE_MPI
