#ifndef CAFFE_CHANNEL_HPP
#define CAFFE_CHANNEL_HPP

#ifdef USE_MPI

#include <boost/shared_ptr.hpp>
using boost::shared_ptr;

namespace caffe {

enum OperationType {
    OP_SUM_ALL, 
    OP_MAX_ALL, 
    OP_GATHER_ALL, 
    OP_GATHER, 
    OP_SCATTER, 
    OP_BROADCAST
};

class MPIJob {
public:
  void* src_ptr_; // str_ptr_==NULL indicates IN_PLACE operation
  void* dst_ptr_;
  int count_;
  int* revcounts_;
  int dtype_size_;
  OperationType op_;
};

class MPIComm{
  public:
    inline static MPIComm& Get() {
      if (!singleton_.get()) {
        singleton_.reset(new MPIComm());
      }
      return *singleton_;
    }

    inline static void DoMPIJob(MPIJob job){ Get().DoJob(job);}

  private:
    MPIComm() {}

    void DoJob(MPIJob &job);

    static shared_ptr<MPIComm> singleton_;
};
};

#endif //USE_MPI

#endif //CAFFE_CHANNEL_HPP
