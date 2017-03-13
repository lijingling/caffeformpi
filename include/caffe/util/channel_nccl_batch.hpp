#ifndef CAFFE_CHANNEL_NCCL_BATCH_HPP
#define CAFFE_CHANNEL_NCCL_BATCH_HPP

#ifdef USE_MPI

#include <boost/shared_ptr.hpp>
#include "caffe/util/channel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "nccl.h"
#include "mpi.h"
using boost::shared_ptr;

namespace caffe {

class NcclBatchComm{
  public:
    inline static NcclBatchComm & Get() {
      if (!singleton_.get()) {
        singleton_.reset(new NcclBatchComm());
      }
      return *singleton_;
    }
    inline static void SetCommMode(int m) {Get().comm_mode_ = m;}
    inline static void SetCommBatchSize(float s) {Get().comm_batch_size_ = s;}
    inline static void AddNcclJob(MPIJob job) { Get().AddJob(job, false);}
    inline static void FlushNcclJob() { Get().FlushJob();}
    inline static void Syncrhonize(){Get().WaitAll();}
    inline static void Destroy(){Get().DestroyAll();}

  private:
    NcclBatchComm():n_runing_job_(0), 
               comm_mode_(SolverParameter_CommMode_LAYER_WISE), 
               comm_batch_size_(10.0f), comm_count_(0) {
      head_job_.src_ptr_ = NULL;
      head_job_.dst_ptr_ = NULL;
      head_job_.count_ = 0; 
    }

    void AddJob(MPIJob& new_job, bool force_to_go);
    void FlushJob();
    void WaitAll();
    void DestroyAll();

    static shared_ptr<NcclBatchComm> singleton_;
    
    std::vector<cudaStream_t> streams_;
    int n_runing_job_;
    int comm_mode_;
    float comm_batch_size_;
    int comm_count_;
    MPIJob head_job_;
};

}

#endif //USE_MPI

#endif //CAFFE_CHANNEL_HPP
