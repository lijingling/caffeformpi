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

namespace caffe {

template <typename Dtype>
RegressionDataLayer<Dtype>::~RegressionDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void RegressionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.regression_data_param().new_height();
  const int new_width  = this->layer_param_.regression_data_param().new_width();
  const bool is_color  = this->layer_param_.regression_data_param().is_color();
  string root_folder = this->layer_param_.regression_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.regression_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.regression_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.regression_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.regression_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.regression_data_param().batch_size();
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

    // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  int dim = this->layer_param_.regression_data_param().fea_dim();
  label_shape.push_back(dim);

  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void RegressionDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void RegressionDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  RegressionDataParameter regression_data_param = this->layer_param_.regression_data_param();
  const int batch_size = regression_data_param.batch_size();
  const int new_height = regression_data_param.new_height();
  const int new_width = regression_data_param.new_width();
  const bool is_color = regression_data_param.is_color();
  string root_folder = regression_data_param.root_folder();
  int fea_dim = regression_data_param.fea_dim();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadDataAugToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    // prefetch_label[item_id] = lines_[lines_id_].second;
    const string separator = "/";
    string path = root_folder + lines_[lines_id_].first;
    
    // similarity to center loss
    vector<string>dest;
    dest.clear();
    split(path, separator, dest);
    int ipos = dest.size();
    string img_name = dest[ipos - 1];
    string lable_name = dest[ipos - 2] + ".bin";
    ipos = path.find_last_of("/");
    string bin_path = path;
    bin_path.erase(ipos,path.size() - ipos);  
    bin_path.append("/");
    bin_path.append(lable_name);
    // LOG(INFO) << "bin_path: " << bin_path;

    
    // autoencoder 
    //path.replace(ipos,4,".bin");

    std::ifstream fin;
    fin.open(bin_path.c_str(), std::ios::binary);
    CHECK(fin.is_open())<< "Could not load " << path;
    float szBuf;
    for (int i = 0; i < fea_dim; ++i){
      fin.read((char*)&szBuf, sizeof(float));
      prefetch_label[item_id*fea_dim  + i ] = szBuf;
      // LOG(INFO) << "szBuf: " << szBuf;
    }
    fin.close();
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      lines_id_ = 0;
      if (this->layer_param_.regression_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
}

INSTANTIATE_CLASS(RegressionDataLayer);
REGISTER_LAYER_CLASS(RegressionData);

}  // namespace caffe
