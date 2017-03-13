#include <algorithm>
#include <vector>
#include <utility>
#include <fstream>
#include <limits> 

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

const int MAX_CODE_LENGTH = 40;

static bool gt(
  const std::pair<unsigned int, unsigned int> &a,
  const std::pair<unsigned int, unsigned int> &b) {
       return a.second > b.second;
}

template <typename Dtype>
inline void SigmoidForward(int n, const Dtype* in, Dtype* out) {
  for(int i = 0; i < n; ++i) {
    out[i] = 1. / (1. + exp(-in[i]));
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::BuildhHuffmanTree(
  const std::string& frequency_hist_fn) {
  unsigned int freq = 0;
  unsigned int idx = 0;
  // load frequency hist from file
  std::vector<std::pair<unsigned int, unsigned int> > frequency_hist; 
  std::ifstream ifs(frequency_hist_fn.c_str());
  if(!ifs.is_open()) {
    if(frequency_hist_fn == "") {
      LOG(WARNING) << "Use default frequency distribution: uniform";
      for(idx = 0; idx < num_output_; ++idx) {
        frequency_hist.push_back(
          std::make_pair<unsigned int, unsigned int>(idx, 1));
      }
    }
    else {
      LOG(FATAL) << "Could not open frequency hist file " << frequency_hist_fn;
    }
  }
  else {
    idx = 0;
    while(ifs >> freq) {
      frequency_hist.push_back(
        std::make_pair<unsigned int, unsigned int>(idx, freq));
      ++idx;
    }
    ifs.close();
  }
  std::sort(frequency_hist.begin(), frequency_hist.end(), gt);
  // validation
  size_t nhist = frequency_hist.size();
  CHECK_EQ(nhist, num_output_);
  LOG(INFO) << "building huffman tree";
  // allocate huffman tree
  huffman_tree_ = (struct huffman_node_t *)calloc(num_output_, sizeof(struct huffman_node_t));
  // Allocate memory for the binary tree construction
  int a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  unsigned int *count = (unsigned int *)calloc(num_output_ * 2 + 1, sizeof(unsigned int));
  unsigned int *binary = (unsigned int *)calloc(num_output_ * 2 + 1, sizeof(unsigned int));
  unsigned int *parent_node = (unsigned int *)calloc(num_output_ * 2 + 1, sizeof(unsigned int));
  for (a = 0; a < num_output_; a++) {
    huffman_tree_[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    huffman_tree_[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
  for (a = 0; a < num_output_; a++) count[a] = frequency_hist[a].second;
  for (a = num_output_; a < num_output_ * 2; a++) count[a] = std::numeric_limits<unsigned int>::max();
  pos1 = num_output_ - 1;
  pos2 = num_output_;
  int max_codelen = 0;
  int min_codelen = MAX_CODE_LENGTH + 1;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < num_output_ - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[num_output_ + a] = count[min1i] + count[min2i];
    parent_node[min1i] = num_output_ + a;
    parent_node[min2i] = num_output_ + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < num_output_; a++) {
    idx = frequency_hist[a].first;
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == num_output_ * 2 - 2) break;
    }
    max_codelen = std::max(i, max_codelen);
    min_codelen = std::min(i, min_codelen);
    huffman_tree_[idx].codelen = i; 
    huffman_tree_[idx].point[0] = num_output_ - 2;
    for (b = 0; b < i; b++) {
      huffman_tree_[idx].code[i - b - 1] = code[b];
      huffman_tree_[idx].point[i - b] = point[b] - num_output_;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
  LOG(INFO) << "finish to build huffman tree, " 
            << "min_codelen = " << min_codelen << ", "
            << "max_codelen = " << max_codelen;
  CHECK_GT(min_codelen, 0);
  CHECK_LE(max_codelen, MAX_CODE_LENGTH)
        << "codelen exceeds MAX_CODE_LENGTH, num_output is too large";
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::DestroyHuffmanTree() {
  if(huffman_tree_) {
    for(int i = 0; i < num_output_; ++i) {
      free(huffman_tree_[i].point);
      free(huffman_tree_[i].code);
    }
    free(huffman_tree_);
  }  
}

template <typename Dtype>
HierarchicalSoftmaxWithLossLayer<Dtype>::~HierarchicalSoftmaxWithLossLayer() {
  DestroyHuffmanTree();
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // vars
  num_output_ = this->layer_param_.hierarchical_softmax_param().num_output();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.hierarchical_softmax_param().axis());
  num_input_ = bottom[0]->count(axis);
  batch_size_ = bottom[0]->count(0, axis);
  const std::string& frequency_hist = 
    this->layer_param_.hierarchical_softmax_param().frequency_hist();
  // build huffman tree
  BuildhHuffmanTree(frequency_hist);
  // Intialize the weight
  this->blobs_.resize(1);
  std::vector<int> weight_shape(2);
  weight_shape[0] = num_output_;
  weight_shape[1] = num_input_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    this->layer_param_.inner_product_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // vars
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.hierarchical_softmax_param().axis());
  batch_size_ = bottom[0]->count(0, axis);  
  int new_num_input = bottom[0]->count(axis);
  CHECK_EQ(num_input_, new_num_input);
  if(bottom.size() == 2) {
    CHECK_EQ(batch_size_, bottom[1]->count()) 
      << "Number of labels must match number of predictions";
  }
  // allocate acid-blob
  std::vector<int> hidden_layer_shape(2);
  hidden_layer_shape[0] = MAX_CODE_LENGTH;
  hidden_layer_shape[1] = num_input_;
  hidden_layer_.Reshape(hidden_layer_shape);
  std::vector<int> sigmoid_product_shape(2);
  sigmoid_product_shape[0] = batch_size_;
  sigmoid_product_shape[1] = MAX_CODE_LENGTH;
  sigmoid_product_.Reshape(sigmoid_product_shape);
  // softmax top shape
  std::vector<int> softmax_shape(1 + axis);
  softmax_shape[0] = batch_size_;
  softmax_shape[axis] = num_output_;
  // determin top shape
  if(bottom.size() == 1) { // without label, only produce prob. distribution
    top[0]->Reshape(softmax_shape);
  }
  else { // with label
    LossLayer<Dtype>::Reshape(bottom, top);
    if (top.size() == 2) {
      top[1]->Reshape(softmax_shape);
    }
  }
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* loss_data = top[0]->mutable_cpu_data();
  Dtype loss = 0;
  Dtype prob;
  for(int i = 0; i < batch_size_; ++i) {
    int label = label_data[i];  
    SigmoidProduct_cpu(bottom[0], i, label);
    prob = Probability_cpu(i, label);
    loss -= log(prob);
  }
  loss_data[0] = loss / batch_size_;
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) {
  Dtype * bottom_diff_data = bottom[0]->mutable_cpu_diff();
  Dtype * syn1_diff_data = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_data);
  caffe_set(this->blobs_[0]->count(), Dtype(0), syn1_diff_data);
  // compuate residual and param grad for each data instance
  const Dtype* label_data = bottom[1]->cpu_data();
  for(int i = 0; i < batch_size_; ++i) {
    ResidualAndGrad_cpu(bottom[0], i, label_data[i]);
  }
  // Scale gradient
  const Dtype loss_weight = top[0]->cpu_diff()[0] / batch_size_;
  caffe_scal(bottom[0]->count(), loss_weight, bottom_diff_data); 
  caffe_scal(this->blobs_[0]->count(), loss_weight, syn1_diff_data);
}

template <typename Dtype>
void HierarchicalSoftmaxWithLossLayer<Dtype>::SigmoidProduct_cpu(
  const Blob<Dtype>* bottom, int data_idx, int label) {
  const Dtype* bottom_data = bottom->cpu_data() + data_idx * num_input_;
  const Dtype* syn1 = this->blobs_[0]->cpu_data();
  Dtype * hidden_layer_data = hidden_layer_.mutable_cpu_data();
  Dtype * sigmoid_product_data = 
    sigmoid_product_.mutable_cpu_data() + data_idx * MAX_CODE_LENGTH;
  const huffman_node_t & label_node = huffman_tree_[label];
  int syn1_offset, hidden_offset;
  // build hidden layer
  for(int d = 0; d < label_node.codelen; ++d) {
    syn1_offset = label_node.point[d] * num_input_;
    hidden_offset = d * num_input_;
    caffe_copy(num_input_, syn1 + syn1_offset, hidden_layer_data + hidden_offset);
  }
  // hidden_layer * x_in
  caffe_cpu_gemv<Dtype>(CblasNoTrans, label_node.codelen, num_input_,
    (Dtype)1., hidden_layer_data, bottom_data, (Dtype)1.,
    sigmoid_product_data);
  // sigmoid
  SigmoidForward(label_node.codelen, sigmoid_product_data, sigmoid_product_data);
}

template <typename Dtype>
Dtype HierarchicalSoftmaxWithLossLayer<Dtype>::Probability_cpu(int data_idx, int label) {
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
void HierarchicalSoftmaxWithLossLayer<Dtype>::ResidualAndGrad_cpu(
    Blob<Dtype>* bottom, int data_idx, int label) {
  Dtype sigmoid, g;
  int syn1_offset;
  Dtype * bottom_diff_data = bottom->mutable_cpu_diff() + data_idx * num_input_;
  Dtype * syn1_diff_data = this->blobs_[0]->mutable_cpu_diff();
  const Dtype * bottom_data = bottom->cpu_data() + data_idx * num_input_;
  const Dtype* syn1_data = this->blobs_[0]->cpu_data();
  const Dtype * sigmoid_product_data = 
    sigmoid_product_.cpu_data() + data_idx * MAX_CODE_LENGTH;
  const huffman_node_t & label_node = huffman_tree_[label];
  
  for(int d = 0; d < label_node.codelen; ++d) {
    syn1_offset = label_node.point[d] * num_input_;
    sigmoid = sigmoid_product_data[d];
    g = label_node.code[d] + sigmoid - 1;
    // residual
    caffe_axpy(num_input_, g, syn1_data + syn1_offset, bottom_diff_data);
    // grad
    caffe_axpy(num_input_, g, bottom_data, syn1_diff_data + syn1_offset);
  }
}

#ifdef CPU_ONLY
STUB_GPU(HierarchicalSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(HierarchicalSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(HierarchicalSoftmaxWithLoss);
}  // namespace caffe
