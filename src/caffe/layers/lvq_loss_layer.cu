#include <vector>
#include <cfloat>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nccl_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}


template <typename Dtype>
__global__ void Compute_positive_distance_gpu(const int M, const int K, const int sub_class, 
  const int label_offset, const Dtype* bottom, const Dtype* label, 
  const Dtype* center, Dtype* distance, Dtype* pos_center) {
  CUDA_KERNEL_LOOP(index, M) {
    const int label_value = label[index] - label_offset;
    if (label_value >= 0 && label_value < sub_class)
    {
      Dtype sum = 0;
      for (int c = 0; c < K; ++c) 
      {
        pos_center[index * K + c] = center[label_value * K + c];
        Dtype var = bottom[index * K + c] - center[label_value * K + c];
        sum += var * var;
      }
      distance[index] = sum;
    }  
    else
    {
      for (int c = 0; c < K; ++c) 
      {
        pos_center[index * K + c] = 0;
      }
      distance[index] = 0;
    }  
  }
}


template <typename Dtype>
__global__ void Lvq_update_bottom_diff_gpu(const int M, const int K, const int sub_class, const int label_offset, 
  const float alpha, const Dtype* bottom, const Dtype* label, const Dtype* center, const Dtype* min_distance, 
  const Dtype* min_label, const Dtype* pos_distance, const Dtype* pos_center, Dtype* bottom_diff, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, M) {
    const int min_label_value = static_cast<int>(min_label[index]);
    //const int label_value = static_cast<int>(label[index]);
    if (min_label_value + label_offset != static_cast<int>(label[index]) 
      && min_distance[index] <= pos_distance[index] + alpha)
    {
      loss[index] = 1;
      for (int i = 0; i < K; ++i)
      {
      bottom_diff[index * K + i] = (center[min_label_value * K + i] - pos_center[index * K + i]);
      }
    }  
    else
    {
      loss[index] = 0;
      for (int i = 0; i < K; ++i)
      {
        bottom_diff[index * K + i] = 0;
      }
    }  
  }
}




template <typename Dtype>
void LvqLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {  
  

  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  Dtype* distance_data = distance_sq_.mutable_gpu_data();

  const int count = distance_sq_.count();
  caffe_gpu_set(M_, Dtype(1), E1_.mutable_gpu_data());
  caffe_gpu_set(sub_class_, Dtype(1), E2_.mutable_gpu_data()); 
  
  //************************************forward***********************************/
  //fast compute distance square
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, K_, 1, bottom_data, 
        bottom_data, bottom_dis_sq_.mutable_gpu_data()); 
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(sub_class_),
      CAFFE_CUDA_NUM_THREADS>>>(sub_class_, K_, 1, center_data, 
        center_data, center_dis_sq_.mutable_gpu_data()); 
  Dtype dot_scaler(-2.0);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, sub_class_, K_, 
    dot_scaler, bottom_data, center_data, (Dtype)0., distance_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, sub_class_, 1, 
    Dtype(1), bottom_dis_sq_.gpu_data(), E2_.gpu_data(), Dtype(1), distance_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, sub_class_, 1, 
    Dtype(1), E1_.gpu_data(), center_dis_sq_.gpu_data(), Dtype(1), distance_data);    

  //compute local min and center index
  caffe_gpu_rmin(sub_class_, M_, distance_data, E1_.mutable_gpu_data(), 
    E1_.mutable_gpu_diff(), block_.mutable_gpu_data(), block_.mutable_gpu_diff());
  

  //caffe_gpu_rmax(sub_class_, M_, distance_data, E1_.mutable_gpu_data(), 
  //  block_.mutable_gpu_data());

  //compute distance to positive center(use bottom_dis_sq_ to store distance)
  Compute_positive_distance_gpu<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, K_, sub_class_, label_offset_, 
        bottom_data, label, center_data, bottom_dis_sq_.mutable_gpu_data(), pos_center_.mutable_gpu_data()); 
  #ifdef USE_MPI
  if(this->need_model_parallel_ && Caffe::parallel_mode() == Caffe::MPI) 
  {
    caffe_nccl_iallreduce(pos_center_.mutable_gpu_data(), pos_center_.mutable_gpu_data(), M_ * K_);
    caffe_nccl_iallreduce(bottom_dis_sq_.mutable_gpu_data(), bottom_dis_sq_.mutable_gpu_data(), M_);
    nccl_force_synchronize();
  }
  #endif

  //for(int i = 0; i < M_; ++i)
  //{
  //  const int label_value = bottom[1]->cpu_data()[i];
  //  std::cout<<i<<" "<<label_value<<" "<<bottom_dis_sq_.cpu_data()[i]<<" "<<E1_.cpu_diff()[i]<<" "<<E1_.cpu_data()[i]<<"\n";
  //}
  //std::cout<<"\nforward*******************************"<<std::endl;


  //Dtype* distance_data = distance_sq_.mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();

  //************************************backward***********************************/
  const bool update_center = this->layer_param_.lvq_loss_param().update_center();
  const float alpha = this->layer_param_.lvq_loss_param().alpha();
  if (!update_center)
  {
    Lvq_update_bottom_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, K_, sub_class_, label_offset_, alpha, bottom_data, label, center_data, 
        E1_.gpu_data(), E1_.gpu_diff(), bottom_dis_sq_.gpu_data(), pos_center_.gpu_data(), bottom_diff, E1_.mutable_gpu_data());
  }   
  else
  {
    NOT_IMPLEMENTED;
  }


  //Dtype dot;
  //caffe_gpu_dot(M_ * K_, bottom_dis_sq_.gpu_data(), bottom_dis_sq_.gpu_data(), &dot);
  //caffe_gpu_asum(M_, bottom_dis_sq_.gpu_data(), &dot);
  //Dtype loss = dot / M_ / Dtype(2);
  //top[0]->mutable_cpu_data()[0] = loss;

  Dtype loss = 0;  
  caffe_gpu_asum(M_, E1_.mutable_gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / M_;

  #ifdef USE_MPI
  if(this->need_model_parallel_ && Caffe::parallel_mode() == Caffe::MPI) 
  {
    caffe_nccl_iallreduce(top[0]->mutable_gpu_data(), top[0]->mutable_gpu_data(), 1);
    caffe_nccl_iallreduce(bottom_diff, bottom_diff, bottom[0]->count());
    nccl_force_synchronize();

    const int rank_size = Caffe::MPI_all_rank();
    top[0]->mutable_cpu_data()[0] /= rank_size;

  }
  #endif

  std::cout<<"**************************************************  "<<top[0]->mutable_cpu_data()[0]<<std::endl;

  Dtype loss_weight = top[0]->cpu_diff()[0]/ M_;
  caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);  
}


template <typename Dtype>
void LvqLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {

  //const Dtype* label = bottom[1]->gpu_data();
  //const Dtype* bottom_data = bottom[0]->gpu_data();
  //const Dtype* center_data = this->blobs_[0]->gpu_data();

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(LvqLossLayer);

}  // namespace caffe
