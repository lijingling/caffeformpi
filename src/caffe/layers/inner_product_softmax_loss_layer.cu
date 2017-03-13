#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nccl_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


/* ------ distributed softmax forward kernal functions ------- */

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
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
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
static void DistSoftmaxForward(
    const int count, const int dim, const int batch_size, 
    const Dtype temprature,
    Dtype* shared_top, Dtype* scale_data, 
    Dtype* block, const Dtype* sum_multiplier) {
  // compute local max
  caffe_gpu_rmax(dim, batch_size, shared_top, scale_data, block);
  // compute global max
  caffe_nccl_iallmax(scale_data, scale_data, batch_size);
  nccl_force_synchronize();
  // subtract
  kernel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, dim, 
    shared_top, scale_data, shared_top);
  if(temprature != (Dtype)1.0) {
    caffe_gpu_scal(count, (Dtype)1.0 / temprature, shared_top);
  }
  // exponentiate
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, shared_top, shared_top);
  // compute local sum after exp
  caffe_gpu_gemv<Dtype>(CblasNoTrans, batch_size, dim, (Dtype)1.,
    shared_top, sum_multiplier, (Dtype)0, scale_data);
  // compute global sum after exp
  caffe_nccl_iallreduce(scale_data, scale_data, batch_size);
  nccl_force_synchronize();
  // divide
  kernel_div<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, dim, scale_data, shared_top); 
}

/* ---------- hard softmax loss kernal functions  ---------- */

template <typename Dtype>
__global__ void HardSoftmaxLossForwardGPU(const int nthreads,
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
__global__ void HardSoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* label, const int label_offset,
          Dtype* bottom_diff, const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]) - label_offset;
    if(label_value >= 0 && label_value < dim) {
      bottom_diff[index * dim + label_value] -= 1;
    }
  }
}

/* ---------- soft softmax loss kernal functions  ---------- */

// we dont need soft softmax loss forward

template <typename Dtype>
__global__ void SoftSoftmaxLossDenseLabelBackwardGPU(
    const int num, const int dim, const int num_output,
    const Dtype* label_data, const int label_offset,
    const Dtype* softmax_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    const int n = index / dim;
    const int s = index % dim;
    const int label_value = s + label_offset;
    bottom_diff[n * dim + s] = 
      softmax_data[n * dim + s] - 
      label_data[n * num_output + label_value];
  }
}

template <typename Dtype>
__global__ void kernal_mean_probility(const int num, const int len,
    const Dtype* in_data, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, num) {
    out_data[index] = (1.0 - in_data[index]) / len;
  }
}

template <typename Dtype>
__global__ void SoftSoftmaxLossSparseLabelBackwardGPU(const int num, 
    const int sparse_dim, const int dense_dim, const Dtype* mean_prob,
    const Dtype* sparse_label, const int label_offset,
    const Dtype* softmax_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, num * sparse_dim) {
    const int n = index / sparse_dim;
    const int s = index % sparse_dim;
    const int n2 = 2 * n;
    const int label_value = sparse_label[n2 * sparse_dim + s] - label_offset;
    if(label_value >= 0 && label_value < dense_dim) {
      const Dtype p = softmax_data[n * dense_dim + label_value];
      const Dtype compensate_p = mean_prob[n];
      const Dtype q = sparse_label[(n2 + 1) * sparse_dim + s];
      bottom_diff[n * dense_dim + label_value] = p - q + compensate_p;
    }
  }
}

template <typename Dtype>
void soft_softmax_loss_sparse_label_backward(
   const int num, const int sparse_dim, 
   const int dim, const int num_output,
   Dtype* scale_data, const Dtype * sparse_prob_sum_multiplier,
   const Dtype* soft_label, const int label_offset,
   Dtype* softmax_data, Dtype* bottom_diff) {
  const int count = num * dim;
  caffe_gpu_gemv<Dtype>(CblasNoTrans, 
    num, 2 * sparse_dim, (Dtype)1., soft_label, 
    sparse_prob_sum_multiplier, (Dtype)0, scale_data);
  kernal_mean_probility<Dtype><<<CAFFE_GET_BLOCKS(num),
    CAFFE_CUDA_NUM_THREADS>>>(num, num_output - sparse_dim, 
    scale_data, scale_data); 
  kernel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, dim, 
    softmax_data, scale_data, softmax_data);
  SoftSoftmaxLossSparseLabelBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(
    num * sparse_dim),
    CAFFE_CUDA_NUM_THREADS>>>(num, sparse_dim, dim, 
    scale_data, soft_label, label_offset, softmax_data, bottom_diff);
}

/* ----------  end of kernal functions  ---------- */

template <typename Dtype>
void InnerProductSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype * bottom_data = bottom[0]->mutable_gpu_data();
  Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype * hard_label = NULL;
  const Dtype * soft_label = NULL;
  Dtype * hard_shared_top = NULL;
  Dtype * soft_shared_top = NULL;
  Dtype hard_temprature, soft_temprature, 
        hard_loss_weight, soft_loss_weight;
  if(hard_bottom_idx_ > 0) {
    hard_label = bottom[hard_bottom_idx_]->gpu_data();
    hard_shared_top = hard_softmax_top_.mutable_gpu_data();
    inner_product_top_.set_gpu_data(hard_shared_top);
    inner_product_top_.set_gpu_diff(hard_shared_top);
    hard_temprature = tempratures_[hard_bottom_idx_ - 1];
    hard_loss_weight = loss_weights_[hard_bottom_idx_ - 1] / 
      bottom[0]->num() / hard_temprature;
  }
  if(soft_bottom_idx_ > 0) {
    soft_label = bottom[soft_bottom_idx_]->gpu_data();
    soft_shared_top = soft_softmax_top_.mutable_gpu_data();
    inner_product_top_.set_gpu_data(soft_shared_top);
    inner_product_top_.set_gpu_diff(soft_shared_top);
    soft_temprature = tempratures_[soft_bottom_idx_ - 1];
    soft_loss_weight = loss_weights_[soft_bottom_idx_ - 1] / 
      bottom[0]->num() / soft_temprature;
  }

  Dtype* inner_product_top = inner_product_top_.mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  const int batch_num = bottom[0]->num() / sub_batch_size_;
  const int sub_batch_count = bottom[0]->count(1) * sub_batch_size_;
  const int dim = inner_product_top_.count(1);
  const int sparse_dim = soft_shared_top ? bottom[soft_bottom_idx_]->count(2) : 0;
  const int count = inner_product_top_.count();
  Dtype sub_loss, loss = 0;
  for(int sub_batch_idx = 0; sub_batch_idx < batch_num; sub_batch_idx++) {  
    // set bottom data
    inner_product_bottom_.set_gpu_data(bottom_data + sub_batch_count * sub_batch_idx);
    inner_product_bottom_.set_gpu_diff(bottom_diff + sub_batch_count * sub_batch_idx);
    
    /*********** forward process ***********/
    
    // inner product layer forward
    inner_product_layer_->Forward(inner_product_bottom_vec_, inner_product_top_vec_);
    // softmax layer forward
    if(hard_shared_top) { // hard softmax forward
      if(inner_product_top != hard_shared_top) {
        caffe_copy(count, inner_product_top, hard_shared_top);
      }
      #ifdef USE_MPI
        if(this->need_model_parallel_ &&
           Caffe::parallel_mode() == Caffe::MPI) {
          DistSoftmaxForward(count, dim, sub_batch_size_, 
            hard_temprature, hard_shared_top, scale_data, 
            block_.mutable_gpu_data(), sum_multiplier_.gpu_data());
        }
        else {
          hard_softmax_layer_->Forward(hard_softmax_bottom_vec_, hard_softmax_top_vec_);
        }
      #else
        hard_softmax_layer_->Forward(hard_softmax_bottom_vec_, hard_softmax_top_vec_);
      #endif
      // loss
      HardSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(sub_batch_size_),
        CAFFE_CUDA_NUM_THREADS>>>(sub_batch_size_, hard_shared_top, 
        hard_label, label_offset_, scale_data, dim);
      caffe_gpu_asum(sub_batch_size_, scale_data, &sub_loss);
      loss += sub_loss;
    }
    if(soft_shared_top) { // soft softmax forward
      if(inner_product_top != soft_shared_top) {
        caffe_copy(count, inner_product_top, soft_shared_top);
      }
      #ifdef USE_MPI
        if(this->need_model_parallel_ &&
           Caffe::parallel_mode() == Caffe::MPI) {
          DistSoftmaxForward(count, dim, sub_batch_size_, soft_temprature,
            soft_shared_top, scale_data, 
            block_.mutable_gpu_data(), sum_multiplier_.gpu_data());
        }
        else {
          soft_softmax_layer_->Forward(soft_softmax_bottom_vec_, soft_softmax_top_vec_);
        }
      #else
        soft_softmax_layer_->Forward(soft_softmax_bottom_vec_, soft_softmax_top_vec_);
      #endif
      // loss
      // we dont report the loss of soft softmax   
    }

    /*********** backward process ***********/

    if(TRAIN == this->phase_) {
      // softmax layer backward
      if(hard_shared_top) { // hard softmax layer backward
        HardSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(sub_batch_size_),
          CAFFE_CUDA_NUM_THREADS>>>(sub_batch_size_, hard_label, label_offset_,
          hard_shared_top, dim);
        caffe_gpu_scal(count, hard_loss_weight, hard_shared_top);
      }
      if(soft_shared_top) { // soft softmax layer backward
        if(!sparse_label_) {
          SoftSoftmaxLossDenseLabelBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(
            sub_batch_size_ * dim),
            CAFFE_CUDA_NUM_THREADS>>>(sub_batch_size_, dim, num_output_, 
            soft_label, label_offset_, soft_shared_top, soft_shared_top);
        }
        else {
          soft_softmax_loss_sparse_label_backward(
            sub_batch_size_, sparse_dim, dim, num_output_, 
            scale_data, sparse_prob_sum_multiplier_.gpu_data(),
            soft_label, label_offset_, soft_shared_top, soft_shared_top);
        }
        caffe_gpu_scal(count, soft_loss_weight, soft_shared_top);
      }
      if(hard_shared_top && soft_shared_top) {
        caffe_gpu_add(count, hard_shared_top, soft_shared_top,
          inner_product_top);
      }
      // inner product backward
      vector<bool> propagate_down(1);
      propagate_down[0] = true;
      inner_product_layer_->Backward(
        inner_product_top_vec_, propagate_down, inner_product_bottom_vec_);
    }
    // go to next sub batch
    if(hard_label) {
      hard_label += sub_batch_size_; 
    }
    if(soft_label) {
      if(!sparse_label_) {
        soft_label += sub_batch_size_ * num_output_;
      }
      else {
        soft_label += sub_batch_size_ * 2 * sparse_dim;
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = loss;
  #ifdef USE_MPI
  if(this->need_model_parallel_ &&
     Caffe::parallel_mode() == Caffe::MPI) {
    // reduce loss
    caffe_nccl_iallreduce(
      top[0]->mutable_gpu_data(), top[0]->mutable_gpu_data(), 1);
    // reduce bottom diff
    if(TRAIN == this->phase_) {
      caffe_nccl_iallreduce(bottom_diff, bottom_diff, bottom[0]->count());
    }
    nccl_force_synchronize();
  }
  #endif
  top[0]->mutable_cpu_data()[0] /= bottom[0]->num();
}

template <typename Dtype>
void InnerProductSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // nothing to do  
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductSoftmaxWithLossLayer);

}  // namespace caffe
