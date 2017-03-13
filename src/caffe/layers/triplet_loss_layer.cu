#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
#include <fstream>
using namespace std;

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename Dtype>
void OnlineTripletLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const Dtype* label = bottom[1]->gpu_data();

         /**
         LOG(INFO) << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
	 LOG(INFO) << "bottom[0]->count(0) = "<<bottom[0]->count(0);
         LOG(INFO) << "bottom[0]->count(1) = "<<bottom[0]->count(1);
         LOG(INFO) << "bottom[0]->count(2) = "<<bottom[0]->count(2);
         LOG(INFO) << "bottom[0]->count(3) = "<<bottom[0]->count(3);
	 LOG(INFO) << "bottom[0]->channels() = "<<bottom[0]->channels();
         LOG(INFO) << "bottom[0]->axis() = "<<bottom[0]->num_axes();
	 LOG(INFO) << bottom[0]->num();**/

	int batch_size = bottom[0]->num();
	int channels = bottom[0]->channels();

	std::vector<int> vShape;

	vShape.clear();
	vShape.push_back(batch_size);
	Blob<Dtype> E1(vShape);
	caffe_gpu_set(E1.count(), Dtype(1.0), E1.mutable_gpu_data());

	vShape.clear();
	vShape.push_back(channels);
	Blob<Dtype> E2(vShape);
	caffe_gpu_set(E2.count(), Dtype(1.0), E2.mutable_gpu_data());

	const Dtype *bottom_data = bottom[0]->gpu_data();
	Dtype *diff_data = diff_.mutable_gpu_data();
	Dtype *dist_ptr = dist_sq_.mutable_gpu_data();
	Dtype *diff_sq_data = diff_sq_.mutable_gpu_data();

	for (int i = 0; i < bottom[0]->num(); i++) {
		caffe_copy(bottom[0]->count(), bottom_data, diff_data + diff_.offset(i
				* batch_size));

		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
				Dtype(1.0), E1.gpu_data(), bottom_data + bottom[0]->offset(i),
				Dtype(-1.0), diff_data + diff_.offset(i * batch_size));

		caffe_gpu_powx(diff_sq_.count(), diff_data + diff_.offset(i
				* batch_size), Dtype(2.0), diff_sq_data);

		caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, batch_size, channels,
				Dtype(1.0), E2.gpu_data(), diff_sq_data, Dtype(0.0), dist_ptr
						+ dist_sq_.offset(i * batch_size));
	}
	// bottom[0]->save("bottom_0.txt");
	// dist_sq_.save("dist_sq_.txt");
	// LOG(INFO) << "SAVE END";
	// getchar();
}

template<typename Dtype>
void OnlineTripletLossLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	// Clear previous gradients.


        const Dtype* label = bottom[1]->cpu_data();
        int num_gpus = this->layer_param_.online_triplet_loss_param().num_gpus();
	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
	int num_anchors = this->layer_param_.online_triplet_loss_param().anchors_per_batch() * num_gpus;
	int num_imgs = this->layer_param_.online_triplet_loss_param().images_per_anchor();
	int num_triplets = num_imgs * (num_imgs - 1) * num_anchors;
	int channels = bottom[0]->channels();
	Dtype margin = this->layer_param_.online_triplet_loss_param().margin();

	int no_semi_hard_found = 0;
	int num_violator = 0;
	Dtype *dist_ptr = dist_sq_.mutable_cpu_data();

        
/**
        ofstream fout("/home/xulinquan/a.txt");
        for(int i=0;i<bottom[1]->num();i++){
	    if(i%5==0){
                fout << "\n";
            }
            fout<<label[i];
            fout<<"   ";
       }
       

       fout <<"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
       fout.close();**/

	for (int id = 0; id < num_anchors; id++) {
		// id picks one person
		for (int i = id * num_imgs; i < (id + 1) * num_imgs; i++) {
			// i picks one anchor image.
			for (int j = id * num_imgs; j < (id + 1) * num_imgs; j++) {
				// j picks one negative image.
				if (i != j) {
					// search for the hard negative
					//LOG(INFO) << "id = "<<id<<"  id = "<<id;
					Dtype dist_ap_sq = *(dist_ptr + dist_sq_.offset(i * bottom[0]->num() + j));
					//LOG(INFO) << "id = "<<id;
					int hard_negative_idx = -1;
					Dtype hard_dist_an_sq = std::numeric_limits<Dtype>::max();
					for (int k = 0; k < bottom[0]->num(); k++) {
						if (label[id*num_imgs]==label[k]) {
                                                    //fout <<"id="<<id;fout<<",k=";fout<<k;fout<<"------------------------------------";fout<<"\n";
							continue;
						}

						// k picks negative image.
						// lookup the distance in dist_sq_.
						Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i* bottom[0]->num() + k));
						if (dist_an_sq > dist_ap_sq && dist_an_sq< hard_dist_an_sq) {
							hard_dist_an_sq = dist_an_sq;
							hard_negative_idx = k;
						}
					}

					if (hard_negative_idx == -1) {
						no_semi_hard_found += 1;
						// Use the one with largest distance instead
						hard_dist_an_sq = -1;
						for (int k = 0; k < bottom[0]->num(); k++) {
							if (label[id*num_imgs]==label[k]) {
                                                             //fout <<"id="<<id;fout<<",k=";fout<<k;fout<<"------------------------------------1111111";fout<<"\n";
								continue;
							}
							Dtype dist_an_sq = *(dist_ptr + dist_sq_.offset(i * bottom[0]->num() + k));
							if (dist_an_sq > hard_dist_an_sq) {
								hard_dist_an_sq = dist_an_sq;
								hard_negative_idx = k;
							}
						}
					}

					CHECK_GE(hard_negative_idx, 0)<< "Error: unlikely bug in finding hard_negative_idx";
					// propagate gradient
					if (margin + dist_ap_sq - hard_dist_an_sq > 0) {
						num_violator += 1;
						int loss_type = this->layer_param_.online_triplet_loss_param().loss_type();
						//LOG(INFO) << "Clear previous gradients";
						Dtype alpha = top[0]->cpu_diff()[0];
						Dtype coefficient = this->layer_param_.online_triplet_loss_param().coefficient();
						Dtype measure;
						switch (loss_type) {
							case 1: {// margin-based conditionallog-likelihoodloss
								measure = coefficient * (hard_dist_an_sq - dist_ap_sq);
								alpha = (1 - sigmoid(-measure));
								break;
							}
							case 2: {//MCE
								measure = coefficient * (hard_dist_an_sq - dist_ap_sq);
								Dtype loss = sigmoid(measure);
								alpha = loss * (1 - loss);
								break;
							}
							case 3: {//GLVQ
								measure = coefficient * (hard_dist_an_sq - dist_ap_sq) / (hard_dist_an_sq - dist_ap_sq);
								Dtype loss = sigmoid(measure);
								alpha = loss * (1 - loss);
								break;
							}
							default: {
								alpha = top[0]->cpu_diff()[0];
								//LOG(INFO) << "alpha =  "<<alpha;
								break;
							}
						}

						// \|\|f(anchor)-f(positive)\|\|_2^2 term, to anchor
						caffe_cpu_axpby(channels, alpha, diff_.cpu_data()+ diff_.offset(i * bottom[0]->num() + j),Dtype(1), bottom[0]->mutable_cpu_diff()+ bottom[0]->offset(i));
						// \|\|f(anchor)-f(negative)\|\|_2^2 term, to anchor
						caffe_cpu_axpby(channels, -alpha, diff_.cpu_data()+ diff_.offset(i * bottom[0]->num()+ hard_negative_idx), Dtype(1),bottom[0]->mutable_cpu_diff()+ bottom[0]->offset(i));
						// \|\|f(anchor)-f(positive)\|\|_2^2 term, to positive
						caffe_cpu_axpby(channels, -alpha, diff_.cpu_data()+ diff_.offset(i * bottom[0]->num() + j),Dtype(1), bottom[0]->mutable_cpu_diff()+ bottom[0]->offset(j));
						// \|\|f(anchor)-f(negative)\|\|_2^2 term, to negative
						caffe_cpu_axpby(channels, alpha, diff_.cpu_data()+ diff_.offset(i * bottom[0]->num()+ hard_negative_idx), Dtype(1),bottom[0]->mutable_cpu_diff()+ bottom[0]->offset(hard_negative_idx));
					}

				}
			}
		}
	}

       // fout.close();
	Dtype coe = 1.0 / Dtype(num_violator + 0.000001);
	caffe_scal(bottom[0]->count(), coe, bottom[0]->mutable_cpu_diff());
	top[0]->mutable_cpu_data()[0] = num_violator / float(num_triplets + 0.000001);
}

INSTANTIATE_LAYER_GPU_FUNCS( OnlineTripletLossLayer);

} // namespace caffe
