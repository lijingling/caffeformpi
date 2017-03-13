//
// Created by alex on 8/25/15.
//

#ifdef USE_MPI

#include "caffe/util/channel.hpp"

#include "caffe/common.hpp"

#include "mpi.h"

namespace caffe {

shared_ptr<MPIComm> MPIComm::singleton_;

void MPIComm::DoJob(MPIJob &job) {
  MPI_Datatype data_type = (job.dtype_size_ == 4) ? MPI_FLOAT : MPI_DOUBLE;

  // call MPI APIs for real works
  switch (job.op_) {
    case OP_SUM_ALL: {
      MPI_CHECK(MPI_Allreduce((job.src_ptr_ == job.dst_ptr_) ? MPI_IN_PLACE : job.src_ptr_,
                              job.dst_ptr_, job.count_, data_type,
                              MPI_SUM, MPI_COMM_WORLD
      ));
      break;
    }
    case OP_MAX_ALL: {
      MPI_CHECK(MPI_Allreduce((job.src_ptr_ == job.dst_ptr_) ? MPI_IN_PLACE : job.src_ptr_,
                              job.dst_ptr_, job.count_, data_type,
                              MPI_MAX, MPI_COMM_WORLD
      ));
      break;
    }
    case OP_GATHER_ALL: {
      MPI_CHECK(MPI_Allgather(job.src_ptr_, job.count_, data_type,
                              job.dst_ptr_, job.count_, data_type,
                              MPI_COMM_WORLD));
      break;
    }
    case OP_GATHER: {
      vector<int> displs(Caffe::MPI_all_rank());
      int displ = 0;
      for(int i = 0; i < displs.size(); ++i) {
        displs[i] = displ;
        displ += job.revcounts_[i];
      }
      MPI_CHECK(MPI_Gatherv(job.src_ptr_, job.count_, data_type,
                           job.dst_ptr_, job.revcounts_, &(displs[0]), 
                           data_type, 0, MPI_COMM_WORLD));
      break;
    }
    case OP_SCATTER: {
      MPI_CHECK(MPI_Scatter(job.src_ptr_, job.count_, data_type,
                            job.dst_ptr_, job.count_, data_type,
                            0, MPI_COMM_WORLD));
      break;
    }
    case OP_BROADCAST: {
      CHECK_EQ(job.src_ptr_, job.dst_ptr_);
      MPI_CHECK(MPI_Bcast(job.src_ptr_, job.count_, data_type,
                          0, MPI_COMM_WORLD));
      break;
    }
    default: {
      LOG(FATAL)<<"Unknown MPI job type";
    }
  }
}

}

#endif //USE_MPI
