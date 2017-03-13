#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nccl_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_subtract(const int count,
    const int cls_num, 
    const Dtype* data, const Dtype* max_val, Dtype * out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / cls_num;
    out[index] = data[index] - max_val[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp_neg(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(-data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_div(const int count,
    const int cls_num, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / cls_num;
    data[index] /= channel_sum[n];
  }
}

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
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const int label_offset,
          Dtype* loss, const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]) - label_offset;
    if(label_value >= 0 && label_value < dim) {
      loss[index] = -log(max(prob_data[index * dim + label_value],
                      Dtype(FLT_MIN)));
    }
    else {
      loss[index] = 0;
    }
  }
}


template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* label, const int label_offset,
          Dtype* bottom_diff, const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]) - label_offset;
    if(label_value >= 0 && label_value < dim) {
      bottom_diff[index * dim + label_value] -= 1;
    }
  }
}


template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const int sub_class, 
  const Dtype* bottom, const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sample_index = index / sub_class;
    const int class_index = index % sub_class;  
    Dtype sum = 0;
    for (int k = 0; k < K; ++k)
    {
      Dtype var = bottom[sample_index * K + k] - center[class_index * K + k];
      sum += var * var;
    }
    distance[index] = sum;
  }
}

template <typename Dtype>
__global__ void BackwardUpdateCenterDiff(const int nthreads, const int M, const int K, 
    const int sub_class, const int label_offset, const Dtype* bottom, const Dtype* label, 
    const Dtype* center, const Dtype* prob_data, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int class_index = index;    
    for (int sample_index = 0; sample_index < M; ++sample_index)
    {
      const int label_value = label[sample_index] - label_offset;
      const int n = sample_index * sub_class + class_index;
      Dtype alpha = 2 * (label_value == class_index ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        center_diff[class_index * K + k] += alpha * (bottom[sample_index * K + k] - center[class_index * K + k]);
      }
    }
  }
}


template <typename Dtype>
__global__ void BackwardUpdateBottomDiff(const int nthreads, const int M, const int K, 
    const int sub_class, const int label_offset, const Dtype* bottom, const Dtype* label, 
    const Dtype* center, const Dtype* prob_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sample_index = index;
    const int label_value = label[sample_index] - label_offset;
    for (int class_index = 0; class_index < sub_class; ++class_index)
    {
      const int n = sample_index * sub_class + class_index;
      Dtype alpha = -2 * (label_value == class_index ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        bottom_diff[sample_index * K + k] += alpha * (bottom[sample_index * K + k] - center[class_index * K + k]);
      }
    }
  }
}


template <typename Dtype>
void SnpcLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {  
  

  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  Dtype* distance_data = distance_sq_.mutable_gpu_data();
  const int dim = distance_sq_.count(1);
  const int count = distance_sq_.count();
  caffe_gpu_set(M_, Dtype(1), E1_.mutable_gpu_data());
  caffe_gpu_set(sub_class_, Dtype(1), E2_.mutable_gpu_data()); 
  caffe_gpu_set(K_, Dtype(1), E3_.mutable_gpu_data()); 

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

  //cudaEvent_t start, stop;
  //float elapsedTime = 0.0;
  //
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start, 0);

  // compute local max
  caffe_gpu_rmax(sub_class_, M_, distance_data, E1_.mutable_gpu_data(), block_.mutable_gpu_data());  
  #ifdef USE_MPI
  if(this->need_model_parallel_ && 
     Caffe::parallel_mode() == Caffe::MPI) {
    // compute global max
    caffe_nccl_iallmax(E1_.mutable_gpu_data(), E1_.mutable_gpu_data(), M_);
    nccl_force_synchronize();
  }
  #endif  
  kernel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, dim, 
    distance_data, E1_.mutable_gpu_data(), distance_data);  
  // exponentiate
  kernel_exp_neg<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, distance_data, distance_data);  
  // compute local sum after exp
  caffe_gpu_gemv<Dtype>(CblasNoTrans, M_, dim, (Dtype)1.,
    distance_data, E2_.gpu_data(), (Dtype)0, E1_.mutable_gpu_data());
  #ifdef USE_MPI
  if(this->need_model_parallel_ && 
     Caffe::parallel_mode() == Caffe::MPI) {
    // compute global sum after exp
    caffe_nccl_iallreduce(E1_.mutable_gpu_data(), E1_.mutable_gpu_data(), M_);
    nccl_force_synchronize();
  }
  #endif
  // divide
  kernel_div<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, dim, E1_.gpu_data(), distance_data);   

  // loss
  Dtype loss = 0;  
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(M_),
    CAFFE_CUDA_NUM_THREADS>>>(M_, distance_data, 
    label, label_offset_, E1_.mutable_gpu_data(), dim);
  caffe_gpu_asum(M_, E1_.mutable_gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss;
  #ifdef USE_MPI
  if(this->need_model_parallel_ && 
    Caffe::parallel_mode() == Caffe::MPI) {
    caffe_nccl_iallreduce(top[0]->mutable_gpu_data(), top[0]->mutable_gpu_data(), 1);
    nccl_force_synchronize();
  }
  #endif
  top[0]->mutable_cpu_data()[0] /= M_;
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //
  //cudaEventElapsedTime(&elapsedTime, start, stop);
  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);
  
  //std::cout<<"**************************************************forward : "<<elapsedTime<<std::endl;
  
}


template <typename Dtype>
void SnpcLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  //cudaEvent_t start, stop;
  //float elapsedTime = 0.0;
  //
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start, 0);

  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  Dtype* distance_data = distance_sq_.mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();

  //Dtype* sum_multiplier_data = sum_multiplier_.mutable_gpu_data();
  //Dtype* sum_multiplier_diff = sum_multiplier_.mutable_gpu_diff();

  //caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  //caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), center_diff);

  /*int nthreads = sub_class_;
  BackwardUpdateCenterDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, sub_class_, label_offset_, 
        bottom_data, label, center_data, distance_data, center_diff); 
  nthreads = M_; 
  BackwardUpdateBottomDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, sub_class_, label_offset_, 
        bottom_data, label, center_data, distance_data, bottom_diff);*/

  //softmax backward  --->   p-delt
  Dtype loss_weight = top[0]->cpu_diff()[0] / M_;
  SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(M_), 
    CAFFE_CUDA_NUM_THREADS>>>(M_, label, label_offset_,
    distance_data, sub_class_);
  caffe_gpu_scal(distance_sq_.count(), loss_weight, distance_data);

  /*for (int i = 0; i < M_ ; ++i)
  {    
    //xi - m
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sub_class_, K_, 1, 
      Dtype(1), E2_.gpu_data(), bottom_data + i * K_, Dtype(0), sum_multiplier_data);
    caffe_gpu_sub<Dtype>(sum_multiplier_.count(), sum_multiplier_data, center_data, sum_multiplier_data);
    //(p-delt)*(xi-m)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sub_class_, K_, 1, 
      Dtype(1), distance_data + i * sub_class_, E3_.gpu_data(), Dtype(0), sum_multiplier_.mutable_gpu_diff());
    caffe_gpu_mul<Dtype>(sum_multiplier_.count(), sum_multiplier_data, sum_multiplier_diff, sum_multiplier_data);
    //update center diff
    caffe_gpu_axpy<Dtype>(sum_multiplier_.count(), Dtype(1), sum_multiplier_data, center_diff);
    //update bottom diff
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, sub_class_, 
      Dtype(-1), E2_.gpu_data(), sum_multiplier_data, Dtype(0), bottom_diff + i * K_);
  }*/
  
  caffe_gpu_set(M_, Dtype(1), E1_.mutable_gpu_data());
  //row sum
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, sub_class_, 
      Dtype(1), distance_data, E2_.gpu_data(), Dtype(0), row_sum_.mutable_gpu_data());
  //col sum
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, sub_class_, M_, 
      Dtype(1), E1_.gpu_data(), distance_data, Dtype(0), col_sum_.mutable_gpu_data());
  //update center diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, sub_class_, K_, 1, 
      Dtype(1), col_sum_.gpu_data(), E3_.gpu_data(), Dtype(0), center_diff);
  caffe_gpu_mul<Dtype>(this->blobs_[0]->count(), center_diff, center_data, center_diff);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, sub_class_, K_, M_, 
      Dtype(1), distance_data, bottom_data, Dtype(-1), center_diff);

  //update bottom diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, 
      Dtype(1), row_sum_.gpu_data(), E3_.gpu_data(), Dtype(0), bottom_diff);
  caffe_gpu_mul<Dtype>(bottom[0]->count(), bottom_diff, bottom_data, bottom_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, sub_class_, 
      Dtype(1), distance_data, center_data, Dtype(-1), bottom_diff);


  #ifdef USE_MPI
  if(this->need_model_parallel_ && Caffe::parallel_mode() == Caffe::MPI) 
  {
	  caffe_nccl_iallreduce(bottom_diff, bottom_diff, bottom[0]->count());
	  nccl_force_synchronize();
  }
  #endif

  caffe_gpu_scal(this->blobs_[0]->count(), Dtype(2), center_diff);
  //Dtype loss_weight = top[0]->cpu_diff()[0] * 2. / M_;
  caffe_gpu_scal(bottom[0]->count(), Dtype(2), bottom_diff);

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //
  //cudaEventElapsedTime(&elapsedTime, start, stop);
  //
  //std::cout<<"**************************************************backward : "<<elapsedTime<<std::endl;
}


INSTANTIATE_LAYER_GPU_FUNCS(SnpcLossLayer);

}  // namespace caffe
