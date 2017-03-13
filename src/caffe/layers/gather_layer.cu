#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nccl_functions.hpp"

namespace caffe {

template <typename Dtype>
void GatherLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {

  #ifdef USE_MPI
  if (Caffe::parallel_mode() == Caffe::MPI){
    for (int i = 0; i < bottom.size(); ++i) {
      // all gather
      caffe_nccl_iallgather((Dtype*)bottom[i]->gpu_data(),
                            (Dtype*)top[i]->mutable_gpu_data(), 
                             bottom[i]->count());
    }
    nccl_force_synchronize();
  }
  #endif
}

template <typename Dtype>
void GatherLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  #ifdef USE_MPI
    if (Caffe::parallel_mode() == Caffe::MPI){
      //Scatter
      for (int i = 0; i < bottom.size(); ++i) {
        if (propagate_down[i]) {
          caffe_nccl_iscatter((Dtype*)top[i]->gpu_diff(),
            (Dtype*)bottom[i]->mutable_gpu_diff(), 
             bottom[i]->count());
        }
      }
      nccl_force_synchronize();
      //compensate the scale on diff IMPORTANT
      for (int i = 0; i < bottom.size(); ++i) {
        if (propagate_down[i]) {
          caffe_gpu_scal(bottom[i]->count(), Dtype(Caffe::MPI_all_rank()),
                         bottom[i]->mutable_gpu_diff());
        }
      }
    }
  #endif
}

INSTANTIATE_LAYER_GPU_FUNCS(GatherLayer);

}  // namespace caffe
