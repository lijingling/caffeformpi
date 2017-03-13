//
// Created by alex on 8/25/15.
//

#ifdef USE_MPI

#include "caffe/caffe.hpp"
#include "caffe/util/mpi_functions.hpp"
#include "caffe/util/channel.hpp"

namespace caffe {
  template <typename Dtype>
  void caffe_mpi_allreduce(Dtype* data, int count){
    MPIJob job = {data, data, count, NULL, sizeof(Dtype), OP_SUM_ALL};
    MPIComm::DoMPIJob(job);
  }

  template void caffe_mpi_allreduce<float>(float* data, int count);
  template void caffe_mpi_allreduce<double>(double* data, int count);

  template <typename Dtype>
  void caffe_mpi_allreduce(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_SUM_ALL};
    MPIComm::DoMPIJob(job);
  }

  template void caffe_mpi_allreduce<float>(float* src_data, float* dst_data, int count);
  template void caffe_mpi_allreduce<double>(double* src_data, double* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_allmax(Dtype* data, int count){
    MPIJob job = {data, data, count, NULL, sizeof(Dtype), OP_MAX_ALL};
    MPIComm::DoMPIJob(job);
  }

  template void caffe_mpi_allmax<float>(float* data, int count);
  template void caffe_mpi_allmax<double>(double* data, int count);

  template <typename Dtype>
  void caffe_mpi_allmax(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_MAX_ALL};
    MPIComm::DoMPIJob(job);
  }

  template void caffe_mpi_allmax<float>(float* src_data, float* dst_data, int count);
  template void caffe_mpi_allmax<double>(double* src_data, double* dst_data, int count);

  template <typename Dtype>
  void caffe_mpi_allgather(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_GATHER_ALL};
    MPIComm::DoMPIJob(job);
  }
  template void caffe_mpi_allgather<float>(float*, float*, int);
  template void caffe_mpi_allgather<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_mpi_gather(Dtype* src_data, Dtype* dst_data, int count, int* revcounts) {
    MPIJob job = {src_data, dst_data, count, revcounts, sizeof(Dtype), OP_GATHER};
    MPIComm::DoMPIJob(job);
  }
  template void caffe_mpi_gather<float>(float*, float*, int, int*);
  template void caffe_mpi_gather<double>(double*, double*, int, int*);

  template <typename Dtype>
  void caffe_mpi_scatter(Dtype* src_data, Dtype* dst_data, int count){
    MPIJob job = {src_data, dst_data, count, NULL, sizeof(Dtype), OP_SCATTER};
    MPIComm::DoMPIJob(job);
  }

  template void caffe_mpi_scatter<float>(float*, float*, int);
  template void caffe_mpi_scatter<double>(double*, double*, int);

  template <typename Dtype>
  void caffe_mpi_bcast(Dtype* data, int count){
    MPIJob job = {data, data, count, NULL, sizeof(Dtype), OP_BROADCAST};
    MPIComm::DoMPIJob(job);
  }
  template void caffe_mpi_bcast<float>(float* data, int count);
  template void caffe_mpi_bcast<double>(double* data, int count);

  void mpi_barrier() {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

#endif //USE_MPI
