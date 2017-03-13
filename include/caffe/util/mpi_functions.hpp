//
// Created by alex on 8/25/15.
//

#ifndef CAFFE_MPI_FUNCTIONS_HPP
#define CAFFE_MPI_FUNCTIONS_HPP

namespace caffe {
  template <typename Dtype>
  void caffe_mpi_allreduce(Dtype* data, int count);

  template <typename Dtype>
  void caffe_mpi_allreduce(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_allmax(Dtype* data, int count);

  template <typename Dtype>
  void caffe_mpi_allmax(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_allgather(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_gather(Dtype* src_data, Dtype* dst_data, int count, int* revcounts);

  template <typename Dtype>
  void caffe_mpi_scatter(Dtype* src_data, Dtype* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_bcast(Dtype* data, int count);
  
  void mpi_barrier();
}

#endif //CAFFE_MPI_FUNCTIONS_HPP_HPP
