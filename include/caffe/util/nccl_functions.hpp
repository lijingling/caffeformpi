//
// Created by alex on 8/25/15.
//

#ifndef CAFFE_NCCL_FUNCTIONS_HPP
#define CAFFE_NCCL_FUNCTIONS_HPP

namespace caffe {
  /* ----------------- nccl collectives ------------------*/ 
  template <typename Dtype>
  void caffe_nccl_iallreduce(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_nccl_iallmax(Dtype* src_data, Dtype* dst_data, int count);
  
  template <typename Dtype>
  void caffe_nccl_iallgather(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_nccl_iscatter(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_nccl_ibcast(Dtype* data, int count);
  
  void nccl_force_synchronize();
  
  /* ----------------- nccl batch collectives ------------------*/ 
  template <typename Dtype>
  void caffe_nccl_batch_iallreduce(Dtype* src_data, Dtype* dst_data, int count);

  void caffe_nccl_batch_flush();
  void nccl_batch_force_synchronize();
}

#endif //CAFFE_MPI_FUNCTIONS_HPP_HPP
