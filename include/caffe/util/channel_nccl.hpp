#ifndef CAFFE_CHANNEL_NCCL_HPP
#define CAFFE_CHANNEL_NCCL_HPP

#ifdef USE_MPI

#include <boost/shared_ptr.hpp>
#include "caffe/util/channel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "nccl.h"
#include "mpi.h"
using boost::shared_ptr;

namespace caffe {

class NcclComm{
  public:
    inline static NcclComm & Get() {
      if (!singleton_.get()) {
        singleton_.reset(new NcclComm());
      }
      return *singleton_;
    }
    inline static void AddNcclJob(MPIJob job) { Get().AddJob(job);}
    inline static void Syncrhonize(){Get().WaitAll();}
    inline static void Destroy(){Get().DestroyAll();}

  private:
    NcclComm(): n_runing_job_(0) {}

    void AddJob(MPIJob& new_job);
    void WaitAll();
    void DestroyAll();

    static shared_ptr<NcclComm> singleton_;
    
    std::vector<cudaStream_t> streams_;
    int n_runing_job_;
};

}

#endif //USE_MPI

#endif //CAFFE_CHANNEL_HPP
