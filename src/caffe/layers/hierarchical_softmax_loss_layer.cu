#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

const int MAX_CODE_LENGTH = 40;

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::SigmoidProduct_gpu(
  const Blob<Dtype>* bottom, int data_idx, int label) {
  const Dtype* bottom_data = bottom->gpu_data() + data_idx * num_input_;
  const Dtype* syn1 = this->blobs_[0]->gpu_data();
  Dtype * hidden_layer_data = hidden_layer_.mutable_gpu_data();
  Dtype * sigmoid_product_data = 
    sigmoid_product_.mutable_gpu_data() + data_idx * MAX_CODE_LENGTH;
  const huffman_node_t & label_node = huffman_tree_[label];
  int syn1_offset, hidden_offset, codelen = label_node.codelen;
  // build hidden layer
  for(int d = 0; d < codelen; ++d) {
    syn1_offset = label_node.point[d] * num_input_;
    hidden_offset = d * num_input_;
    caffe_copy(num_input_, syn1 + syn1_offset, hidden_layer_data + hidden_offset);
  }
  // hidden_layer * x_in
  caffe_gpu_gemv<Dtype>(CblasNoTrans, codelen, num_input_,
    (Dtype)1., hidden_layer_data, bottom_data, (Dtype)1.,
    sigmoid_product_data);
  // sigmoid
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(codelen), CAFFE_CUDA_NUM_THREADS>>>(
      codelen, sigmoid_product_data, sigmoid_product_data);
}

template <typename Dtype>
Dtype HierarchicalSoftmaxWithLossLayer<Dtype>::Probability_gpu(int data_idx, int label) {
  Dtype prob = 1, sigmoid;
  const Dtype * sigmoid_product_data = 
    sigmoid_product_.cpu_data() + data_idx * MAX_CODE_LENGTH;
  const huffman_node_t & label_node = huffman_tree_[label];
  for(int d = 0; d < label_node.codelen; ++d) {
    sigmoid = sigmoid_product_data[d];
    prob *= label_node.code[d] == 0 ? sigmoid : 1 - sigmoid;
  }
  return prob;
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::ResidualAndGrad_gpu(
    Blob<Dtype>* bottom, int data_idx, int label) {
  Dtype sigmoid, g;
  int syn1_offset;
  Dtype * bottom_diff_data = bottom->mutable_gpu_diff() + data_idx * num_input_;
  Dtype * syn1_diff_data = this->blobs_[0]->mutable_gpu_diff();
  const Dtype * bottom_data = bottom->gpu_data() + data_idx * num_input_;
  const Dtype* syn1_data = this->blobs_[0]->gpu_data();
  const Dtype * sigmoid_product_data = 
    sigmoid_product_.cpu_data() + data_idx * MAX_CODE_LENGTH;
  const huffman_node_t & label_node = huffman_tree_[label];
  
  for(int d = 0; d < label_node.codelen; ++d) {
    syn1_offset = label_node.point[d] * num_input_;
    sigmoid = sigmoid_product_data[d];
    g = label_node.code[d] + sigmoid - 1;
    // residual
    caffe_gpu_axpy(num_input_, g, syn1_data + syn1_offset, bottom_diff_data);
    // grad
    caffe_gpu_axpy(num_input_, g, bottom_data, syn1_diff_data + syn1_offset);
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* loss_data = top[0]->mutable_cpu_data();
  Dtype loss = 0;
  for(int i = 0; i < batch_size_; ++i) {
    SigmoidProduct_gpu(bottom[0], i, label_data[i]);
  }
  for(int i = 0; i < batch_size_; ++i) {
    loss -= log(Probability_gpu(i, label_data[i]));
  }
  loss_data[0] = loss / batch_size_;
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) {
  Dtype * bottom_diff_data = bottom[0]->mutable_gpu_diff();
  Dtype * syn1_diff_data = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff_data);
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), syn1_diff_data);
  // compuate residual and param grad for each data instance
  const Dtype* label_data = bottom[1]->cpu_data();
  for(int i = 0; i < batch_size_; ++i) {
    ResidualAndGrad_gpu(bottom[0], i, label_data[i]);
  }
  // Scale gradient
  const Dtype loss_weight = top[0]->cpu_diff()[0] / batch_size_;
  caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff_data); 
  caffe_gpu_scal(this->blobs_[0]->count(), loss_weight, syn1_diff_data);
}


INSTANTIATE_LAYER_GPU_FUNCS(HierarchicalSoftmaxWithLossLayer);

}  // namespace caffe