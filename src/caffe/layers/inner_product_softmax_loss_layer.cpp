#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void InnerProductSoftmaxWithLossLayer<Dtype>::LayerSetUp(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  label_offset_ = 0;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  num_input_ = bottom[0]->count(axis);  
  num_output_ = this->layer_param_.inner_product_param().num_output(); 

  // figure out index of hard and soft label
  hard_bottom_idx_ = -1;
  soft_bottom_idx_ = -1;
  if(1 == bottom[1]->num_axes()) {
    hard_bottom_idx_ = 1;
  }
  else {
    soft_bottom_idx_ = 1;
  }
  if(3 == bottom.size()) {
    if(1 == bottom[2]->num_axes()) {
      hard_bottom_idx_ = 2;
    }
    else {
      soft_bottom_idx_ = 2;
    }
  }
  CHECK_NE(hard_bottom_idx_, soft_bottom_idx_);
  
  // figure out temprature and loss weights
  const HardSoftSoftmaxParameter& hard_soft_param = 
    this->layer_param_.hard_soft_softmax_param();
  sparse_label_ = hard_soft_param.sparse_label();
  loss_weights_.resize(bottom.size() - 1, (Dtype)1.0);
  tempratures_.resize(bottom.size() - 1, (Dtype)1.0);
  if (hard_soft_param.loss_weight_size() != 0) {
    CHECK_EQ(hard_soft_param.loss_weight_size(), bottom.size() - 1); 
    for(int i = 0; i < bottom.size() - 1; ++i) {
      loss_weights_[i] = hard_soft_param.loss_weight(i);
    }
  }
  if(hard_soft_param.temprature_size() != 0) {
    CHECK_EQ(hard_soft_param.temprature_size(), bottom.size() - 1); 
    for(int i = 0; i < bottom.size() - 1; ++i) {
      tempratures_[i] = hard_soft_param.temprature(i);
    }
  }
  if(2 == bottom.size()) { // ignore hard_soft_param if there is only one label bottom
    loss_weights_[0] = this->layer_param_.loss_weight_size() == 0 ? 
      (Dtype)1.0 : this->layer_param_.loss_weight(0);
    tempratures_[0] = this->layer_param_.softmax_param().temprature();
    sparse_label_ = this->layer_param_.softmax_param().sparse_label();
  }

  // figure out sub batch size
  sub_batch_size_ = this->layer_param_.sub_batch_size();
  sub_batch_size_ = std::max(sub_batch_size_, 1);
  sub_batch_size_ = std::min(sub_batch_size_, bottom[0]->num());
  CHECK(bottom[0]->num() % sub_batch_size_ == 0) 
    << "sub_batch_size must be divisible by batch_size(" 
    << bottom[0]->num() << ")";

  // shape bottom
  std::vector<int> inner_product_bottom_shape = bottom[0]->shape();
  inner_product_bottom_shape[0] = sub_batch_size_;
  inner_product_bottom_.Reshape(inner_product_bottom_shape);

  /******* setup inner product layer *******/
  LOG(INFO) << "Creating nested inner product layer"; 
  LayerParameter inner_product_param(this->layer_param_);
  inner_product_param.set_type("InnerProduct");
  #ifdef USE_MPI
    if(this->need_model_parallel_ &&
       Caffe::parallel_mode() == Caffe::MPI) {     
      // figure out num_ouput for this slice
      int rank_size = Caffe::MPI_all_rank();
      int my_rank = Caffe::MPI_my_rank();
      CHECK_GE(num_output_, rank_size);
      int sub_num_output = num_output_ / rank_size;
      if(my_rank == rank_size - 1) {
        sub_num_output += num_output_ % rank_size;
      }
      // set filter type to constant temporarily
      InnerProductParameter * mutable_inner_product_param = 
        inner_product_param.mutable_inner_product_param();
      mutable_inner_product_param->set_num_output(sub_num_output);
      mutable_inner_product_param->mutable_weight_filler()->set_type("constant");
      mutable_inner_product_param->mutable_bias_filler()->set_type("constant");
    }
  #endif
  inner_product_layer_ = LayerRegistry<Dtype>::CreateLayer(inner_product_param);
  inner_product_bottom_vec_.clear();
  inner_product_bottom_vec_.push_back(&inner_product_bottom_);
  inner_product_top_vec_.clear();
  inner_product_top_vec_.push_back(&inner_product_top_);
  inner_product_layer_->SetUp(inner_product_bottom_vec_, inner_product_top_vec_);
  LOG(INFO) << "Top shape: " << inner_product_top_.shape_string(); 
  #ifdef USE_MPI
    if(this->need_model_parallel_ &&
       Caffe::parallel_mode() == Caffe::MPI) {
      // now, filter inner product layer weights
      const int rank_size = Caffe::MPI_all_rank();
      const int my_rank = Caffe::MPI_my_rank();
      label_offset_ = (num_output_ / rank_size) * my_rank;
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().weight_filler()));
      weight_filler->Fill(inner_product_layer_->blobs()[0].get(), 
          label_offset_ * num_input_, num_output_ * num_input_);
      bool bias_term = this->layer_param_.inner_product_param().bias_term();
      if(bias_term) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
        bias_filler->Fill(inner_product_layer_->blobs()[1].get(), 
          label_offset_, num_output_);
      }
    }
  #endif

  /******* setup hard softmax layer *******/

  if(hard_bottom_idx_ > 0) {
    LOG(INFO) << "Creating nested hard softmax layer"; 
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_param.mutable_softmax_param()->
      set_temprature(tempratures_[hard_bottom_idx_ - 1]);
    hard_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    hard_softmax_bottom_vec_.clear();
    hard_softmax_bottom_vec_.push_back(&inner_product_top_);
    hard_softmax_top_vec_.clear();
    hard_softmax_top_vec_.push_back(&hard_softmax_top_);
    hard_softmax_layer_->SetUp(hard_softmax_bottom_vec_, hard_softmax_top_vec_);
    LOG(INFO) << "Top shape: " << hard_softmax_top_.shape_string(); 
  }
 
  /******* setup soft softmax layer *******/
  if(soft_bottom_idx_ > 0) {
    LOG(INFO) << "Creating nested soft softmax layer"; 
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_param.mutable_softmax_param()->
      set_temprature(tempratures_[soft_bottom_idx_ - 1]);
    softmax_param.mutable_softmax_param()->
      set_sparse_label(sparse_label_);
    soft_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    soft_softmax_bottom_vec_.clear();
    soft_softmax_bottom_vec_.push_back(&inner_product_top_);
    soft_softmax_top_vec_.clear();
    soft_softmax_top_vec_.push_back(&soft_softmax_top_);
    soft_softmax_layer_->SetUp(soft_softmax_bottom_vec_, soft_softmax_top_vec_);
    LOG(INFO) << "Top shape: " << soft_softmax_top_.shape_string(); 
  }
  
  // shallow copy inner product weights to this
  vector<shared_ptr<Blob<Dtype> > >& inner_product_blobs = 
     inner_product_layer_->blobs();
  this->blobs_.resize(inner_product_blobs.size());
  for(int i = 0; i < this->blobs_.size(); ++i) {
    this->blobs_[i].reset(new Blob<Dtype>());
    this->blobs_[i]->ReshapeLike(*(inner_product_blobs[i]));
    this->blobs_[i]->ShareData(*(inner_product_blobs[i]));
    this->blobs_[i]->ShareDiff(*(inner_product_blobs[i]));
  }
 
  // shape additional vars for soft softmax 
  if(soft_bottom_idx_ > 0 && sparse_label_) {
    const int sparse_dim = bottom[soft_bottom_idx_]->count(2);
    vector<int> sum_shape(1, 2 * sparse_dim);
    sparse_prob_sum_multiplier_.Reshape(sum_shape);
    Dtype * multiplier = sparse_prob_sum_multiplier_.mutable_cpu_data();
    caffe_set(sparse_dim, Dtype(0), multiplier);
    caffe_set(sparse_dim, Dtype(1), multiplier + sparse_dim);
  }
  // shape additional vars for model-parallel mode
  #ifdef USE_MPI
    if(this->need_model_parallel_ &&
       Caffe::parallel_mode() == Caffe::MPI) {
      this->dist_blobs_.resize(this->blobs_.size());
      this->dist_blob_counts_.resize(this->blobs_.size());
      // shape global weights
      for(int i = 0; i < this->dist_blobs_.size(); ++i) {
        this->dist_blobs_[i].reset(new Blob<Dtype>());
        std::vector<int> dist_blob_shape = this->blobs_[i]->shape();
        dist_blob_shape[0] = num_output_;
        this->dist_blobs_[i]->Reshape(dist_blob_shape);
        const int rank_size = Caffe::MPI_all_rank();
        for(int my_rank = 0; my_rank < rank_size; ++my_rank) {
          int rank_count = 
            (my_rank != rank_size - 1) ? 
            (num_output_ / rank_size) : 
            (num_output_ / rank_size + num_output_ % rank_size);
          this->dist_blob_counts_[i].push_back(
            i == 0 ? num_input_ * rank_count : rank_count);
        }
      }
      // shape block for caffe_gpu_rmax
      vector<int> block_shape(2);
      block_shape[0] = sub_batch_size_;
      block_shape[1] =  
        (inner_product_top_.shape(1) / COMPATIBLE_CUDA_NUM_THREADS) + 1;
      block_.Reshape(block_shape);
      // need sum multiplier
      vector<int> sum_shape(1, inner_product_top_.shape(1));
      sum_multiplier_.Reshape(sum_shape);
      caffe_set(sum_shape[0], Dtype(1), sum_multiplier_.mutable_cpu_data());
    }
  #endif  
}

template <typename Dtype>
void InnerProductSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top); 
  if(hard_bottom_idx_ > 0) {
    CHECK_EQ(bottom[hard_bottom_idx_]->num_axes(), 1);
    CHECK_EQ(bottom[0]->num(), bottom[hard_bottom_idx_]->count()) << 
      "label count must be equal to batch size";
  }
  if(soft_bottom_idx_ > 0) {
    if(!sparse_label_) {
      CHECK_EQ(bottom[0]->num() * num_output_, 
        bottom[soft_bottom_idx_]->count()) <<
        "soft softmax layer inputs must have the same count.";
    }
    else {
      CHECK_EQ(bottom[soft_bottom_idx_]->shape(1), 2) <<
       " layer inputs must have 2 channels in sparse mode";
    }
  }
  // figure out sub batch size
  sub_batch_size_ = this->layer_param_.sub_batch_size();
  sub_batch_size_ = std::max(sub_batch_size_, 1);
  sub_batch_size_ = std::min(sub_batch_size_, bottom[0]->num());
  CHECK(bottom[0]->num() % sub_batch_size_ == 0) 
    << "sub_batch_size must be divisible by batch_size(" 
    << bottom[0]->num() << ")";
  // shape bottom
  std::vector<int> inner_product_bottom_shape = bottom[0]->shape();
  inner_product_bottom_shape[0] = sub_batch_size_;
  inner_product_bottom_.Reshape(inner_product_bottom_shape); 
  // reshape inner product layer
  inner_product_layer_->Reshape(inner_product_bottom_vec_, inner_product_top_vec_);
  // reshape hard softmax layer 
  if(hard_bottom_idx_ > 0) {
    hard_softmax_layer_->Reshape(hard_softmax_bottom_vec_, hard_softmax_top_vec_);
  }
  // reshape soft softmax layer 
  if(soft_bottom_idx_ > 0) {
    soft_softmax_layer_->Reshape(soft_softmax_bottom_vec_, soft_softmax_top_vec_);
  }
  // other vars
  vector<int> scale_shape(1);
  scale_shape[0] = sub_batch_size_;
  scale_.Reshape(scale_shape);
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(InnerProductSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(InnerProductSoftmaxWithLoss);

}  // namespace caffe
