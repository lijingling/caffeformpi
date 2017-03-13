#ifdef USE_MPI

#include "caffe/caffe.hpp"
#include "caffe/util/nccl_functions.hpp"
#include "caffe/util/channel_nccl.hpp"
#include "caffe/util/channel_nccl_batch.hpp"

namespace caffe {
  /* ----------------- nccl collectives ------------------*/
  template <typename Dtype>
  void caffe_nccl_iallreduce(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_SUM_ALL};
    NcclComm::AddNcclJob(job);
  }

  template void caffe_nccl_iallreduce<float>(float* src_data, float* dst_data, int count);
  template void caffe_nccl_iallreduce<double>(double* src_data, double* dst_data, int count);

  template <typename Dtype>
  void caffe_nccl_iallmax(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_MAX_ALL};
    NcclComm::AddNcclJob(job);
  }

  template void caffe_nccl_iallmax<float>(float* src_data, float* dst_data, int count);
  template void caffe_nccl_iallmax<double>(double* src_data, double* dst_data, int count);

  template <typename Dtype>
  void caffe_nccl_iallgather(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_GATHER_ALL};
    NcclComm::AddNcclJob(job);
  }
  template void caffe_nccl_iallgather<float>(float*, float*, int);
  template void caffe_nccl_iallgather<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_nccl_iscatter(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_SCATTER};
    NcclComm::AddNcclJob(job);
  }

  template void caffe_nccl_iscatter<float>(float*, float*, int);
  template void caffe_nccl_iscatter<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_nccl_ibcast(Dtype* data, int count){
    MPIJob job = {data, data, count, NULL, sizeof(Dtype), OP_BROADCAST};
    NcclComm::AddNcclJob(job);
  }
  template void caffe_nccl_ibcast<float>(float* data, int count);
  template void caffe_nccl_ibcast<double>(double* data, int count);
  
  void nccl_force_synchronize(){
    NcclComm::Syncrhonize();
  }

/* ----------------- nccl batch collectives ------------------*/
  template <typename Dtype>
  void caffe_nccl_batch_iallreduce(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_SUM_ALL};
    NcclBatchComm::AddNcclJob(job);
  }

  template void caffe_nccl_batch_iallreduce<float>(float* src_data, float* dst_data, int count);
  template void caffe_nccl_batch_iallreduce<double>(double* src_data, double* dst_data, int count);
  
  void caffe_nccl_batch_flush() {
    NcclBatchComm::FlushNcclJob();
  }

  void nccl_batch_force_synchronize(){
    NcclBatchComm::Syncrhonize();
  }
}

#endif //USE_MPI
