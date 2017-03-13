#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SnpcLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    label_offset_ = 0;
    const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.snpc_loss_param().axis());
    K_ = bottom[0]->count(axis);
    const int num_output = this->layer_param_.snpc_loss_param().num_output();
    N_ = num_output;
    M_ = bottom[0]->num();
    sub_class_ = N_;

    if (this->blobs_.size() > 0) 
    {
       LOG(INFO) << "Skipping parameter initialization";
    } 
    else 
    {
        #ifdef USE_MPI
        if(this->need_model_parallel_ && Caffe::parallel_mode() == Caffe::MPI) 
        {
            // now, filter inner product layer weights
            const int rank_size = Caffe::MPI_all_rank();
            const int my_rank = Caffe::MPI_my_rank();
            
            sub_class_ = N_ / rank_size;
            if(my_rank == rank_size - 1) 
            {
            	sub_class_ += N_ % rank_size;
            }
            label_offset_ = (N_ / rank_size) * my_rank;
        }
        #endif
        	
        // Intialize the weight
        this->blobs_.resize(1);
        vector<int> center_shape(2);
        center_shape[0] = sub_class_;
        center_shape[1] = K_;
        this->blobs_[0].reset(new Blob<Dtype>(center_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.snpc_loss_param().center_filler())); 
        center_filler->Fill(this->blobs_[0].get(), label_offset_ * K_, N_ * K_);  
        
        #ifdef USE_MPI
        if(this->need_model_parallel_ && Caffe::parallel_mode() == Caffe::MPI) 
        {
            const int rank_size = Caffe::MPI_all_rank();
            //const int my_rank = Caffe::MPI_my_rank();
            // allocate memory for dist_blob
            this->dist_blobs_.resize(this->blobs_.size());
            this->dist_blob_counts_.resize(this->blobs_.size());
            for(int i = 0; i < this->dist_blobs_.size(); ++i) 
            {
                this->dist_blobs_[i].reset(new Blob<Dtype>());
                std::vector<int> dist_blob_shape = this->blobs_[i]->shape();
                dist_blob_shape[0] = N_;
                this->dist_blobs_[i]->Reshape(dist_blob_shape);
                for(int rank = 0; rank < rank_size; ++rank) 
                {
                    int rank_count = (rank != rank_size - 1) ? (N_ / rank_size) : (N_ / rank_size + N_ % rank_size);	
                    //std::cout<<rank<<" "<<rank_count<<std::endl;
                    this->dist_blob_counts_[i].push_back(rank_count * K_);
                }
            }
        }        
        #endif       
        this->param_propagate_down_.resize(this->blobs_.size(), true);        
    }
}


template <typename Dtype>
void SnpcLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    // The top shape will be the bottom shape with the flattened axes dropped,
    // and replaced by a single axis with dimension num_output (N_).
    LossLayer<Dtype>::Reshape(bottom, top);
    E1_.Reshape(M_, 1, 1, 1);
    E2_.Reshape(sub_class_, 1, 1, 1);   
    E3_.Reshape(K_, 1, 1, 1);
    row_sum_.Reshape(M_, 1, 1, 1);
    col_sum_.Reshape(sub_class_, 1, 1, 1);

    distance_sq_.Reshape(M_, sub_class_, 1, 1);
    bottom_dis_sq_.Reshape(M_, 1, 1, 1);
    center_dis_sq_.Reshape(sub_class_, 1, 1, 1);

    sum_multiplier_.Reshape(sub_class_, K_, 1, 1);

    
    vector<int> block_shape(2);
    block_shape[0] = M_;
    block_shape[1] =  (sub_class_ / COMPATIBLE_CUDA_NUM_THREADS) + 1;
    block_.Reshape(block_shape);
}
#ifdef CPU_ONLY
STUB_GPU(SnpcLossLayer);
#endif

INSTANTIATE_CLASS(SnpcLossLayer);
REGISTER_LAYER_CLASS(SnpcLoss);

}  // namespace caffe


