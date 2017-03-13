#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/algorithm/string/split.hpp>

#include <boost/algorithm/string.hpp>

using namespace std;
namespace caffe {

template <typename Dtype>
TripletDataLayer<Dtype>::~TripletDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void TripletDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.triplet_data_param().new_height();
  const int new_width  = this->layer_param_.triplet_data_param().new_width();
  const bool is_color  = this->layer_param_.triplet_data_param().is_color();
  string root_folder = this->layer_param_.triplet_data_param().root_folder();
  const int num_classes  = this->layer_param_.triplet_data_param().num_classes();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  bool is_triplet = this->layer_param_.triplet_data_param().is_triplet();
  std::vector<std::string> list_names;
  const string& source = this->layer_param_.triplet_data_param().source();
  boost::split(list_names, source, boost::is_any_of(","));
  img_idx.clear();
  img_idx.resize(list_names.size());
  img_lines.clear();
  img_lines.resize(list_names.size());
  for(int i = 0; i < list_names.size(); i++)
  {
    img_lines[i].clear();
    LOG(INFO) << "Opening file " << list_names[i];
    std::ifstream infile(list_names[i].c_str());
    string filename;
    int label;
    while (infile >> filename >> label) {
    img_lines[i].push_back(std::make_pair(filename, label));
    }
    infile.close();
    LOG(INFO) << "A total of " << img_lines[i].size() << " images of "<<list_names[i].c_str();
    int lines_size = img_lines[i].size();
    if (this->phase_ == TRAIN&&is_triplet) 
    {
      labels_filenames.clear();
      std::map<int, vector<std::string> >::iterator iter;      
      for(int item_id = 0; item_id < lines_size; ++item_id)
      {
        int label_id = img_lines[i][item_id].second;
        iter = labels_filenames.find(label_id);
        if(iter == labels_filenames.end())
        {
          vector<std::string> filenames(1,img_lines[i][item_id].first);
          labels_filenames.insert(std::pair<int,vector<std::string> >(label_id,filenames));
        }
        else
        {
          iter->second.push_back(img_lines[i][item_id].first);
        }
      }

      img_idx[i].clear();
      img_idx[i].resize(num_classes);

      for(int item_id = 0; item_id < img_idx[i].size(); ++item_id)
      {
        img_idx[i][item_id].clear();    
      }

      for(int item_id = 0; item_id < lines_size; ++item_id)
      {


        int label_id = img_lines[i][item_id].second;
        img_idx[i][label_id].push_back(item_id); 
      }
      people_size = num_classes;
    }
  }


  if (this->phase_ == TRAIN&&is_triplet) {
	  //loaded source of negative
	  const string& source_negative = this->layer_param_.triplet_data_param().source_negative();
	  std::ifstream infile_negative(source_negative.c_str());
	  string filename_n;
	  int label_n;
	  while (infile_negative >> filename_n >> label_n) {
		  img_lines_negative.push_back(std::make_pair(filename_n, label_n));
	  }
	  infile_negative.close();
	  LOG(INFO) << "A total of " << img_lines_negative.size() << " images of negative source";
  }


  lines_id_ = 0;
  negative_lines_id_ = 0;

  rand_idx.resize(people_size);
  for (int i = 0; i < people_size; ++i)
  {
    rand_idx[i] = i;
  }

  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();

  string img_path = img_lines[0][0].first;

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + img_path, new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void TripletDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(rand_idx.begin(), rand_idx.end(), prefetch_rng);
  shuffle(img_lines_negative.begin(), img_lines_negative.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TripletDataLayer<Dtype>::InternalThreadEntry() {
  static boost::mt19937 rng(time(0)); 
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  TripletDataParameter triplet_data_param = this->layer_param_.triplet_data_param();
  const int batch_size = triplet_data_param.batch_size();
  const int new_height = triplet_data_param.new_height();
  const int new_width = triplet_data_param.new_width();
  const bool is_color = triplet_data_param.is_color();
  string root_folder = triplet_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.

  string img_path = img_lines[0][0].first;
  cv::Mat cv_img = ReadImageToCVMat(root_folder + img_path, new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  // datum scales
  // const int lines_size = lines_.size();

 // datum scales
  boost::uniform_int<> ui(0, img_lines.size() - 1); 
  int source_id = ui(rng);
  if (source_id == img_lines.size()) source_id = img_lines.size() -1;

  const int lines_size = img_lines[0].size();

  bool is_triplet = triplet_data_param.is_triplet() ;

  int num_anchors = triplet_data_param.num_anchors() ;
  int num_pos_imgs = triplet_data_param.num_pos_imgs() ;
  int num_negative = triplet_data_param.num_negative();
  vector<int>tmp_idx(batch_size);
  tmp_idx.clear();
  vector<int>anchors_idx(num_anchors);
  anchors_idx.clear(); 
  if (this->phase_ == TRAIN&&is_triplet)
  {
    int nValidAnchors = 0;
    while(nValidAnchors < num_anchors) 
    {
      int people_id = rand_idx[lines_id_];
      lines_id_ ++;
      if (lines_id_ >= people_size){
    	  lines_id_ = 0;
        ShuffleImages();
      }

      if(img_idx[source_id][people_id].size() < num_pos_imgs)
      {
        continue;
      }
      
      ++nValidAnchors;
      anchors_idx.push_back(people_id);
      std::random_shuffle(img_idx[source_id][people_id].begin(), img_idx[source_id][people_id].end());
      for (int j = 0; j < num_pos_imgs; ++j)
      {
        int t_idx = img_idx[source_id][people_id][j] ;
        CHECK_GT(lines_size, t_idx);
        tmp_idx.push_back(t_idx);
      }

    }


    for(int i=0 ; i<num_negative; i++){
    	negative_lines_id_ ++;
    	if(negative_lines_id_ >= img_lines_negative.size()){
    		negative_lines_id_ = 0;
    	}

    	tmp_idx.push_back(negative_lines_id_);
    }

  }





  int line_t = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    if (this->phase_ == TRAIN&&is_triplet)
    {
    	line_t = tmp_idx[item_id];
    }

    cv::Mat cv_img;
    if(item_id < num_anchors*num_pos_imgs){
    	CHECK_GT(lines_size, line_t);
    	cv_img = ReadDataAugToCVMat(root_folder + img_lines[source_id][line_t].first,
    	        new_height, new_width, is_color);

    	CHECK(cv_img.data) << "Could not load " << img_lines[source_id][line_t].first;
    }else{
    	cv_img = ReadDataAugToCVMat(root_folder + img_lines_negative[line_t].first,
    	    	        new_height, new_width, is_color);
    }

    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    // prefetch_label[item_id] = lines_[lines_id_].second;
    if(item_id < num_anchors*num_pos_imgs){
    	prefetch_label[item_id] = img_lines[source_id][line_t].second;
    }else{
    	prefetch_label[item_id] = img_lines_negative[line_t].second;
    }
  }
}

INSTANTIATE_CLASS(TripletDataLayer);
REGISTER_LAYER_CLASS(TripletData);

}  // namespace caffe
