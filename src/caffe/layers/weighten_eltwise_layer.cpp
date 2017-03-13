#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightenEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
  || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
  "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
  == EltwiseParameter_EltwiseOp_PROD
  && this->layer_param().eltwise_param().coeff_size())) <<
  "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
 
  if (op_ == EltwiseParameter_EltwiseOp_SUM)
  {
    this->blobs_.resize(1);
    // this->blobs_[0]->Reshape(bottom.size(), 1, 1, 1);
    this->blobs_[0].reset(new Blob<Dtype>(bottom.size(), 1, 1, 1));
    caffe_set(this->blobs_[0]->count(), Dtype(1), this->blobs_[0]->mutable_cpu_data()); 
  }
  else
  {
    coeffs_ = vector<Dtype>(bottom.size(), 1);
    if (this->layer_param().eltwise_param().coeff_size()) {
      for (int i = 0; i < bottom.size(); ++i) {
        coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
      }
    }
  }

  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

template <typename Dtype>
void WeightenEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}

template <typename Dtype>
void WeightenEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  const Dtype* mutable_coeff = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);       //c[i]=a[i]*b[i]
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_axpy(count, mutable_coeff[i], bottom[i]->cpu_data(), top_data);             //b[i]=alph*a[i]+b[i]
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();         //¡¤????¨²¡ä?¡ê? top_data¡ä?¡ä¡é¦Ì?¨º?¨ª?¨°?????¦Ì?¡Á?¡ä¨®?¦Ì
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid              //mask¡ä?¦Ì?¨º??¨²¦Ì¨²????blob¨¤???¨¨?¦Ì?¡Á?¡ä¨®?¦Ì
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void WeightenEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);      //3?¨º??¡¥ bottom_diff
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff); 
        }
    /*
    y=x1*x2*...*xn
    div(L)/div(xi)=div(L)/div(y)*div(y)/div(xi)=div(L)/div(y)*x1...x(i-1)*x(i+1)...xn
    */
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);       
        break;
      case EltwiseParameter_EltwiseOp_SUM:
      {
        Dtype* mutable_coeff_diff = this->blobs_[0]->mutable_cpu_diff();
        const Dtype* mutable_coeff_data = this->blobs_[0]->cpu_data();
        caffe_cpu_scale(count, mutable_coeff_data[i], top_diff, bottom_diff);
      // LOG(INFO)<<"mutable_coeff_old:"<<mutable_coeff_data[i]<<endl;
        mutable_coeff_diff[i] = caffe_cpu_dot(count, top_diff, bottom_data)/Dtype(bottom[0]->num());       
      //  LOG(INFO)<<"mutable_coeff:"<<mutable_coeff_diff[i]<<endl;
        break;
      }
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];    //gradient = top_diff[index]?
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightenEltwiseLayer);
#endif

INSTANTIATE_CLASS(WeightenEltwiseLayer);
REGISTER_LAYER_CLASS(WeightenEltwise);

}  // namespace caffe
