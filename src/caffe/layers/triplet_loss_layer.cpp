#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // TODO: sanity check for anchor_per_batch and images_per_anchor
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
 // int num_anchors = this->layer_param_.online_triplet_loss_param().anchors_per_batch();
  //int num_pos_imgs = this->layer_param_.online_triplet_loss_param().images_per_anchor();
  //int num_imgs = bottom[0]->num();
  int batch_size = bottom[0]->num();
  diff_.Reshape(batch_size * batch_size, bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(batch_size, bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(batch_size * batch_size, 1, 1, 1);
  
}

template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::Reshape(
              const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom.size(), 2) << "OnlineTripletLossLayer takes two bottom input";
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.    
    top[0]->Reshape(loss_shape);
}
    
template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //int num_anchors = this->layer_param_.online_triplet_loss_param().anchors_per_batch();
  //int num_pos_imgs = this->layer_param_.online_triplet_loss_param().images_per_anchor();
  int channels = bottom[0]->channels();
  
  // calculate pairwise diff
  // TODO: test how slow this is
  const Dtype *data_ptr = bottom[0]->cpu_data();
  Dtype *diff_ptr = diff_.mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); i++) {
      // i picks one anchor image.
      for (int j = 0; j < bottom[0]->num(); j++) {
    // j picks from other anchor or negative images.
    caffe_sub(
        channels,
        data_ptr + bottom[0]->offset(i),
        data_ptr + bottom[0]->offset(j),
        diff_ptr + diff_.offset(i * bottom[0]->num() + j));
      }
  }
  
  // calculate the pairwise distance
  Dtype *dist_ptr = dist_sq_.mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); i++) {
      // i picks one anchor image.
      for (int j = 0; j < bottom[0]->num(); j++) {
    // j picks from other anchor or negative images.
    *(dist_ptr+dist_sq_.offset(i * bottom[0]->num() + j)) =
        caffe_cpu_dot(channels,
          diff_ptr + diff_.offset(i * bottom[0]->num() + j),
          diff_ptr + diff_.offset(i * bottom[0]->num() + j));
      }
  }

  // calculate the average loss
  // TODO: report top_k loss
  // Dtype margin = this->layer_param_.online_triplet_loss_param().margin();
  // //Dtype loss(0.0);
  // for (int id = 0; id < num_anchors; id++) {
  //     // id picks one person
  //     for (int i = id*num_pos_imgs; i < (id+1)*num_pos_imgs; i++) {
   //  // i picks one anchor image.
   //  for (int j = id*num_pos_imgs; j < (id+1)*num_pos_imgs; j++) {
   //  // j picks postive image.
   //      if (i != j) {
    //   Dtype dist_ap_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+j));
    //   for (int k = 0; k < bottom[0]->num();k++) {

    //    if ((k > id*num_pos_imgs - 1)&&(k < (id+1)*num_pos_imgs))
    //    {
    //      continue;
    //    }

    //       // k picks negative image.
    //       // lookup the distance in dist_sq_.
    //     Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+k));
    //     loss += std::max(margin+dist_ap_sq-dist_an_sq, Dtype(0));
    //   }
   //      }
   //  }
  //     }
  // }

  // // The effective number of triplets.
  // int num_triplets = num_pos_imgs*(num_pos_imgs-1)*num_anchors;
  
  // loss = loss / static_cast<Dtype>(num_triplets);
  // top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    // Clear previous gradients.
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

    int num_gpus = this->layer_param_.online_triplet_loss_param().num_gpus();
    int num_anchors = this->layer_param_.online_triplet_loss_param().anchors_per_batch() * num_gpus;
    int num_imgs = this->layer_param_.online_triplet_loss_param().images_per_anchor();
    int num_triplets = num_imgs*(num_imgs-1)*num_anchors;
    int channels = bottom[0]->channels();
    Dtype margin = this->layer_param_.online_triplet_loss_param().margin();
    
    int no_semi_hard_found = 0;
    int num_violator = 0;
    //Dtype loss(0.0);
    //LOG(INFO) << "bottom[0]->num()" << bottom[0]->num();
    Dtype *dist_ptr = dist_sq_.mutable_cpu_data();


    for (int id = 0; id < num_anchors; id++) {
      // id picks one person
      for (int i = id*num_imgs; i < (id+1)*num_imgs; i++) {
    // i picks one anchor image.
    for (int j = id*num_imgs; j < (id+1)*num_imgs; j++) {
        // j picks one negative image.
        if (i!=j) {
      // search for the hard negative
      Dtype dist_ap_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+j));
      int hard_negative_idx = -1;
      Dtype hard_dist_an_sq = std::numeric_limits<Dtype>::max();
      for (int k = 0; k < bottom[0]->num();k++) 
      {
        if ((k > id*num_imgs - 1)&&(k < (id+1)*num_imgs))
        {
          continue;
        }
        // k picks negative image.
        // lookup the distance in dist_sq_.
        Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+k));
        if (dist_an_sq > dist_ap_sq && dist_an_sq < hard_dist_an_sq) 
        {
        hard_dist_an_sq = dist_an_sq;
        hard_negative_idx = k;
        }
      }

      if (hard_negative_idx == -1) {
          no_semi_hard_found += 1;
          // Use the one with largest distance instead
          hard_dist_an_sq = -1;
          for (int k = 0; k < bottom[0]->num();k++) {
            if ((k > id*num_imgs - 1)&&(k < (id+1)*num_imgs))
          {
            continue;
          }

        Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+k));
        if (dist_an_sq > hard_dist_an_sq) {
            hard_dist_an_sq = dist_an_sq;
            hard_negative_idx = k;
        }
          }
      }

      CHECK_GE(hard_negative_idx, 0) << "Error: unlikely bug in finding hard_negative_idx";
      // propagate gradient
      if (margin+dist_ap_sq-hard_dist_an_sq > 0) 
      {
          num_violator += 1;
        
      }
        }
    }
      }
    }

    top[0]->mutable_cpu_data()[0] = num_violator/float(num_triplets);

  //LOG(INFO) << "num_violator = " << num_violator << " num_triplets = "<<num_triplets;






    for (int id = 0; id < num_anchors; id++) {
      // id picks one person
      for (int i = id*num_imgs; i < (id+1)*num_imgs; i++) {
    // i picks one anchor image.
    for (int j = id*num_imgs; j < (id+1)*num_imgs; j++) {
        // j picks one negative image.
        if (i!=j) {
      // search for the hard negative
      Dtype dist_ap_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+j));
      int hard_negative_idx = -1;
      Dtype hard_dist_an_sq = std::numeric_limits<Dtype>::max();
      for (int k = 0; k < bottom[0]->num();k++) 
      {
        if ((k > id*num_imgs - 1)&&(k < (id+1)*num_imgs))
        {
          continue;
        }
        // k picks negative image.
        // lookup the distance in dist_sq_.
        Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+k));
        if (dist_an_sq > dist_ap_sq && dist_an_sq < hard_dist_an_sq) 
        {
        hard_dist_an_sq = dist_an_sq;
        hard_negative_idx = k;
        }
      }

      if (hard_negative_idx == -1) {
          no_semi_hard_found += 1;
          // Use the one with largest distance instead
          hard_dist_an_sq = -1;
          for (int k = 0; k < bottom[0]->num();k++) {
            if ((k > id*num_imgs - 1)&&(k < (id+1)*num_imgs))
          {
            continue;
          }

        Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i*bottom[0]->num()+k));
        if (dist_an_sq > hard_dist_an_sq) {
            hard_dist_an_sq = dist_an_sq;
            hard_negative_idx = k;
        }
          }
      }

      CHECK_GE(hard_negative_idx, 0) << "Error: unlikely bug in finding hard_negative_idx";
      // propagate gradient
      if (margin+dist_ap_sq-hard_dist_an_sq > 0) {
          const Dtype alpha = top[0]->cpu_diff()[0] / Dtype(num_violator);

          // \|\|f(anchor)-f(positive)\|\|_2^2 term, to anchor
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data()+diff_.offset(i*bottom[0]->num()+j),
              Dtype(1),
              bottom[0]->mutable_cpu_diff()+bottom[0]->offset(i));
          // \|\|f(anchor)-f(negative)\|\|_2^2 term, to anchor
          caffe_cpu_axpby(
              channels,
              -alpha,
              diff_.cpu_data()+diff_.offset(i*bottom[0]->num()+hard_negative_idx),
              Dtype(1),
              bottom[0]->mutable_cpu_diff()+bottom[0]->offset(i));
          // \|\|f(anchor)-f(positive)\|\|_2^2 term, to positive
          caffe_cpu_axpby(
              channels,
              -alpha,
              diff_.cpu_data()+diff_.offset(i*bottom[0]->num()+j),
              Dtype(1),
              bottom[0]->mutable_cpu_diff()+bottom[0]->offset(j));
          // \|\|f(anchor)-f(negative)\|\|_2^2 term, to negative
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data()+diff_.offset(i*bottom[0]->num()+hard_negative_idx),
              Dtype(1),
              bottom[0]->mutable_cpu_diff()+bottom[0]->offset(hard_negative_idx));
      }
        }
    }
      }
    }



    Dtype coefficient = this->layer_param_.online_triplet_loss_param().coefficient();
    
    caffe_scal(bottom[0]->count(),coefficient,bottom[0]->mutable_cpu_diff());

    
    //if (no_semi_hard_found > 0) {
  //LOG(WARNING) << "semi-hard-negative cannot be found on "
  //       << no_semi_hard_found << " triplets.";
    //}


    //LOG(INFO) << "Margin Violation Rate" << num_violator/float(num_triplets);
}
#ifdef CPU_ONLY
STUB_GPU(OnlineTripletLossLayer);
#endif

INSTANTIATE_CLASS(OnlineTripletLossLayer);
REGISTER_LAYER_CLASS(OnlineTripletLoss);

}  // namespace caffe
