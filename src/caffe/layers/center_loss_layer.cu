#include <vector>
#include <ctime>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

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
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label, const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    // distance(i) = x(i) - c_{y(i)}
    distance[index] = bottom[index] - center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, Dtype* variation_sum, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= distance[m * K + k];
        }
      }
    }
    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = variation_sum[index * K + k] /(count + (Dtype)1.);
    }
  }
}


template <typename Dtype>
__global__ void Compute_distance_data_gpu_2(int nthreads, const int K, const int sub_class, 
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
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;


  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  Dtype* distance_data = distance_sq_.mutable_gpu_data();
  const int dim = distance_sq_.count(1);
  const int count = distance_sq_.count();
  caffe_gpu_set(M_, Dtype(1), E1_.mutable_gpu_data());
  caffe_gpu_set(N_, Dtype(1), E2_.mutable_gpu_data()); 

  //fast compute distance square
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, K_, 1, bottom_data, 
        bottom_data, bottom_dis_sq_.mutable_gpu_data()); 
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(N_),
      CAFFE_CUDA_NUM_THREADS>>>(N_, K_, 1, center_data, 
        center_data, center_dis_sq_.mutable_gpu_data()); 
  Dtype dot_scaler(-2.0);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, 
    dot_scaler, bottom_data, center_data, (Dtype)0., distance_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, 
    Dtype(1), bottom_dis_sq_.gpu_data(), E2_.gpu_data(), Dtype(1), distance_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, 
    Dtype(1), E1_.gpu_data(), center_dis_sq_.gpu_data(), Dtype(1), distance_data);  

  //for (int i = 0; i < 50; ++i)
  //{
  //  std::cout<<distance_sq_.cpu_data()[i]<<" ";
  //}
  //std::cout<<"\n************************************"<<std::endl;

  //Compute_distance_data_gpu_2<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, N_, bottom[0]->gpu_data(),
  //      this->blobs_[0]->gpu_data(), distance_sq_.mutable_gpu_data());
//
  // compute local max
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, N_, 1, distance_data, E1_.mutable_gpu_data()); 
  //caffe_gpu_rmax(N_, M_, distance_data, E1_.mutable_gpu_data(), block_.mutable_gpu_data());
  // subtract
  kernel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, dim, 
    distance_data, E1_.mutable_gpu_data(), distance_sq_.mutable_gpu_data()); 
  // exponentiate
  kernel_exp_neg<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, distance_data, distance_sq_.mutable_gpu_data());   
     
  // compute local sum after exp
  caffe_gpu_gemv<Dtype>(CblasNoTrans, M_, dim, (Dtype)1.,
    distance_data, E2_.gpu_data(), (Dtype)0, E1_.mutable_gpu_data());
  // divide
  kernel_div<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, dim, E1_.mutable_gpu_data(), distance_sq_.mutable_gpu_data());   
  // loss
  loss = 0;  
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(M_),
    CAFFE_CUDA_NUM_THREADS>>>(M_, distance_data, 
    label, 0, E1_.mutable_gpu_data(), dim);
  //std::cout<<E1_.cpu_data()[0]<<std::endl;
  //std::cout<<distance_sq_.cpu_data()[int(bottom[1]->cpu_data()[0])]<<std::endl;
  caffe_gpu_asum(M_, E1_.mutable_gpu_data(), &loss);
  //std::cout<<"********************************************loss: "<<loss / M_<<std::endl;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //int nthreads = N_;
  //caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
  //Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //    CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), 
  //                              variation_sum_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());

  if (propagate_down[0]) {
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
