#include <vector>
#include <cfloat>
#include <ctime>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
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
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_scale(const int count, const Dtype scale, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = data[index] * scale;
  }
}

template <typename Dtype>
__global__ void kernel_exp_neg(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(-data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
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
__global__ void DistanceLossForwardGPU(const int nthreads, const int num_class_total,
    const int num_img_per_class, const Dtype* prob_data, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / num_img_per_class;
    loss[index] = -log(max(prob_data[index * num_class_total + n], Dtype(FLT_MIN)));
    }
}

/*
template <typename Dtype>
__global__ void CenterDistanceLossBackwardGPU(const int nthreads, const int M, const int K, 
    const int num_class_total, const int num_img_per_class, const Dtype* variation, 
    const Dtype* prob_data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int m = index / num_class_total;//the image id
    const int n = index % num_class_total;//the class id
    if (m / num_img_per_class == n)
    {
      for (int i = 0; i < K; ++i)
      {
        out[index * K + i] = (prob_data[m * num_class_total + n] - 1) * (-2) * variation[index * K + i];
      }
    }
    else
    {
      for (int i = 0; i < K; ++i)
      {
        out[index * K + i] = (prob_data[m * num_class_total + n]) * (-2) * variation[index * K + i];
      }
    }
    }
}*/


template <typename Dtype>
__global__ void BackwardUpdateCenterDiff(const int nthreads, const int M, const int K, 
    const int num_class_total, const int num_img_per_class, const Dtype* variation, 
    const Dtype* label, const Dtype* prob_data, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int m = index;// the m-th class
    const int label_value = label[m * num_img_per_class];
    for (int i = 0; i < M; ++i)
    {
      const int n = i * num_class_total + m;
      Dtype alpha = 2 * (i / num_img_per_class == m ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        center_diff[label_value * K + k] += alpha * variation[n * K + k];
      }
    }
  }
}


template <typename Dtype>
__global__ void BackwardUpdateCenterDiff_1(const int nthreads, const int M, const int K, 
    const int num_class_total, const int num_img_per_class, const Dtype* bottom, 
    const Dtype* label, const Dtype* center, const Dtype* prob_data, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int class_index = index;
    const int label_value = label[class_index * num_img_per_class];
    for (int sample_index = 0; sample_index < M; ++sample_index)
    {
      const int n = sample_index * num_class_total + class_index;
      Dtype alpha = 2 * (sample_index / num_img_per_class == class_index ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        center_diff[label_value * K + k] += alpha * (bottom[sample_index * K + k] - center[label_value * K + k]);
      }
    }
  }
}


template <typename Dtype>
__global__ void BackwardUpdateBottomDiff(const int nthreads, const int M, const int K, 
    const int num_class_total, const int num_img_per_class, const Dtype* variation, 
    const Dtype* label, const Dtype* prob_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int m = index;// the m-th sample
    for (int i = 0; i < num_class_total; ++i)
    {
      const int n = m * num_class_total + i;
      Dtype alpha = -2 * (m / num_img_per_class == i ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        bottom_diff[m * K + k] += alpha * variation[n * K + k];
      }
    }
  }
}


template <typename Dtype>
__global__ void BackwardUpdateBottomDiff_1(const int nthreads, const int M, const int K, 
    const int num_class_total, const int num_img_per_class, const Dtype* bottom, 
    const Dtype* label, const Dtype* center, const Dtype* prob_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sample_index = index;
    for (int class_index = 0; class_index < num_class_total; ++class_index)
    {
      const int label_value = label[class_index * num_img_per_class];
      const int n = sample_index * num_class_total + class_index;
      Dtype alpha = -2 * (sample_index / num_img_per_class == class_index ? (prob_data[n] - 1) : prob_data[n]);

      for (int k = 0; k < K; ++k)
      {
        bottom_diff[sample_index * K + k] += alpha * (bottom[sample_index * K + k] - center[label_value * K + k]);
      }
    }
  }
}


/*
template <typename Dtype>
__global__ void Compute_variation_data_gpu(int nthreads, const int K, const int num_class_total,
  const int num_img_per_class, const Dtype* bottom, const Dtype* label, const Dtype* center, Dtype* variation, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i = 0; i < num_class_total; ++i)
    {
      Dtype sum = 0;
      const int n = index * num_class_total + i;
      const int label_value = static_cast<int>(label[i * num_img_per_class]);
      for (int j =0; j < K; ++j)
      {
        Dtype var = bottom[index * K + j] - center[label_value * K + j];        
        variation[n * K + j] = var;
        sum += var * var;
      }
      distance[n] = sum;
    }
  }
}*/

template <typename Dtype>
__global__ void Compute_variation_data_gpu_1(int nthreads, const int M, const int K, const int num_class_total,
  const int num_img_per_class, const Dtype* bottom, const Dtype* label, const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sample_index = index / num_class_total;
    const int class_index = index % num_class_total;
    const int label_value = static_cast<int>(label[class_index * num_img_per_class]);
    Dtype sum = 0;
    for (int k = 0; k < K; ++k)
    {
      Dtype var = bottom[sample_index * K + k] - center[label_value * K + k];
      //variation[index * K + k] = var;
      sum += var * var;
    }
    distance[sample_index * num_class_total + class_index] = sum;
  }
}

/*
template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const int num_class_total,
  const Dtype* variation, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / num_class_total;
    int k = index % num_class_total;
    Dtype sum = 0;
    for (int i = 0; i < K; ++i)
    {
      sum += variation[index * K + i] * variation[index * K + i];
    }
    distance[m * num_class_total + k] = sum;
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
}*/


template <typename Dtype> 
void DistanceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Dtype* variation_data = variation_.mutable_gpu_data();
  Dtype* distance_data = distance_.mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();

  //cudaEvent_t start, stop;
  //float time = 0.0f;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start, 0);
  /*int nthreads = M_;
  Compute_variation_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, num_class_total_, num_img_per_class_, bottom[0]->gpu_data(), 
        bottom[1]->gpu_data(), this->blobs_[0]->gpu_data(), variation_data, distance_data);*/
  int nthreads = M_ * num_class_total_;
  Compute_variation_data_gpu_1<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, num_class_total_, num_img_per_class_, bottom[0]->gpu_data(), 
        bottom[1]->gpu_data(), this->blobs_[0]->gpu_data(), distance_data);

  //compute distance
  /*nthreads = M_ * num_class_total_;
  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, num_class_total_, variation_.gpu_data(), distance_data);*/

  //caffe_gpu_powx(distance_.count(), distance_data, Dtype(1./ 2), distance_data);
  //finish = clock();
  //printf("forward : %f\n", (float)(finish - start));
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  nthreads = M_;
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(M_, num_class_total_, 1, distance_data, scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  nthreads = M_ * num_class_total_;
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, num_class_total_, 1, scale_data, distance_data);
  //caffe_gpu_scal(nthreads, Dtype(-1), distance_data);

  // exponentiate(-1)
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp_neg<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, distance_data, distance_data);
  /*for (int i = 0; i < 5; ++i)
  {
    std::cout<<distance_.cpu_data()[i]<<" ";
  }
  std::cout<<"\n***************************************************"<<std::endl;*/
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  nthreads = M_;
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(M_, num_class_total_, 1, distance_data, scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  nthreads = M_ * num_class_total_;
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, num_class_total_, 1, scale_data, distance_data); 
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&time, start, stop);
  //std::cout<<"******************************************************forward : "<<time<<std::endl; 

  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  nthreads = M_;
  DistanceLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_class_total_, num_img_per_class_, distance_data, loss_data);  

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= M_;
  top[0]->mutable_cpu_data()[0] = loss;
  //std::cout<<"*****************************************"<<loss<<std::endl;
}


template <typename Dtype>
void DistanceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  //const Dtype* variation_data = variation_.gpu_data();
  const Dtype* distance_data = distance_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), center_diff);

  //cudaEvent_t start, stop;
  //float time = 0.0f;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start, 0);

  if (propagate_down[0]) {
  //update center_diff
  int nthreads = num_class_total_; 
  /*BackwardUpdateCenterDiff_1<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, num_class_total_, num_img_per_class_, 
        variation_data, label, distance_data, center_diff);*/
  BackwardUpdateCenterDiff_1<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, num_class_total_, num_img_per_class_, 
        bottom_data, label, center_data, distance_data, center_diff);
  //update bottom_diff
  nthreads = M_; 
  /*BackwardUpdateBottomDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, num_class_total_, num_img_per_class_, 
        variation_data, label, distance_data, bottom_diff);*/
  BackwardUpdateBottomDiff_1<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, num_class_total_, num_img_per_class_, 
        bottom_data, label, center_data, distance_data, bottom_diff);
  caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1./M_), center_diff);
  caffe_gpu_scal(bottom[0]->count(), Dtype(1./M_), bottom_diff);
  }
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&time, start, stop);
  //std::cout<<"******************************************************backward : "<<time<<std::endl;


  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DistanceLossLayer);

}  // namespace caffe
