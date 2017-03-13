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
TriRegDataLayer<Dtype>::~TriRegDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void TriRegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.trireg_data_param().new_height();
  const int new_width  = this->layer_param_.trireg_data_param().new_width();
  const bool is_color  = this->layer_param_.trireg_data_param().is_color();
  string root_folder = this->layer_param_.trireg_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  bool is_triplet = this->layer_param_.trireg_data_param().is_triplet();
  std::vector<std::string> list_names;
  const string& source = this->layer_param_.trireg_data_param().source();
  boost::split(list_names, source, boost::is_any_of(","));
  img_idx.clear();
  img_idx.resize(list_names.size());
  img_lines.clear();
  img_lines.resize(list_names.size());

  // LOG(INFO) << "Opening file " << list_names.size();
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
      //LOG(INFO) << "item_id" << item_id << " images.";
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
      img_idx[i].resize(labels_filenames.size());

      for(int item_id = 0; item_id < img_idx[i].size(); ++item_id)
      {
        img_idx[i][item_id].clear();    
      }


      for(int item_id = 0; item_id < lines_size; ++item_id)
      {
        int label_id = img_lines[i][item_id].second;
        img_idx[i][label_id].push_back(item_id); 
      }    
    }
  }



  lines_id_ = 0;
  people_size = labels_filenames.size();

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
  const int batch_size = this->layer_param_.trireg_data_param().batch_size();
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
    
  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  fea_dim = this->layer_param_.regression_data_param().fea_dim() + 1;
  label_shape.push_back(fea_dim);

  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void TriRegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(rand_idx.begin(), rand_idx.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TriRegDataLayer<Dtype>::InternalThreadEntry() {
  static boost::mt19937 rng(time(0)); 
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  TriRegDataParameter trireg_data_param = this->layer_param_.trireg_data_param();
  const int batch_size = trireg_data_param.batch_size();
  const int new_height = trireg_data_param.new_height();
  const int new_width = trireg_data_param.new_width();
  const bool is_color = trireg_data_param.is_color();
  string root_folder = trireg_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.

  string img_path = img_lines[0][0].first;
  cv::Mat cv_img = ReadDataAugToCVMat(root_folder + img_path, new_height, new_width, is_color);
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
  //LOG(INFO) << "source_id = "<<source_id<<"   img_lines.size() = "<<img_lines.size(); 
  if (source_id == img_lines.size()) source_id = img_lines.size() -1;

  //int source_id = ui(rng);
  const int lines_size = img_lines[source_id].size();
  //LOG(ERROR) << "lines_size = "<<lines_size;  
   
  bool is_triplet = trireg_data_param.is_triplet() ;

  int num_anchors = trireg_data_param.num_anchors() ;
  int num_pos_imgs = trireg_data_param.num_pos_imgs() ;
  vector<int>tmp_idx(batch_size);
  tmp_idx.clear();
  vector<int>anchors_idx(num_anchors);
  anchors_idx.clear(); 

  if (this->phase_ == TRAIN&&is_triplet)
  {
    int nValidAnchors = 0;
    while(nValidAnchors < num_anchors) 
    {
      // boost::uniform_int<> img_idx_ui(0, img_idx[source_id].size() - 1);
      lines_id_ ++;
      int people_id = rand_idx[lines_id_];
      //people_id = people_id % people_size;
      if (lines_id_ >= people_size){
        lines_id_ = 0;
        ShuffleImages();
        // std::random_shuffle(img_idx[0].begin(), img_idx[0].end());
      }

      if(img_idx[source_id][people_id].size() < num_pos_imgs)
      {
        continue;
      }
      
//      if( anchors_idx.end() != find(anchors_idx.begin(), anchors_idx.end(), people_line_id))
//      {
//        continue;
//      }
      ++nValidAnchors;
      // anchors_idx.push_back(people_id);

      std::random_shuffle(img_idx[source_id][people_id].begin(), img_idx[source_id][people_id].end());
      for (int j = 0; j < num_pos_imgs; ++j)
      {
        int t_idx = img_idx[source_id][people_id][j] ;
        CHECK_GT(lines_size, t_idx);
        tmp_idx.push_back(t_idx);
      }

    }


    int item_num = tmp_idx.size();

    while(item_num < batch_size)
    {
      boost::uniform_int<> rand_ui(0, lines_size - 1); 
      int item_id_tmp = rand_ui(rng);  
      if (item_id_tmp == lines_size) item_id_tmp = lines_size -1; 
      bool is_same = false;
      for (int j = 0; j < anchors_idx.size(); ++j)
      {
        if(img_lines[source_id][item_id_tmp].second == anchors_idx[j])
        {
          is_same = true;
        }
      } 

      if (!is_same)
      {
        tmp_idx.push_back(item_id_tmp);
        item_num++;
      }
    }

    //ShuffleImages(img_idx);
    // 
  }


  int line_t = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    if (this->phase_ == TRAIN&&is_triplet)
    {
      line_t = tmp_idx[item_id];
    }
    CHECK_GT(lines_size, line_t);

    cv::Mat cv_img = ReadDataAugToCVMat(root_folder + img_lines[source_id][line_t].first,
        new_height, new_width, is_color);

    CHECK(cv_img.data) << "Could not load " << img_lines[source_id][line_t].first;
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    // prefetch_label[item_id] = lines_[lines_id_].second;
    prefetch_label[item_id*fea_dim] = img_lines[source_id][line_t].second;

     const string separator = "/";
    string path = root_folder + img_lines[source_id][line_t].first;
    

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
    
    std::ifstream fin;
    fin.open(bin_path.c_str(), std::ios::binary);
    CHECK(fin.is_open())<< "Could not load " << path;
    float szBuf;
    for (int i = 1; i < fea_dim + 1; ++i){
      fin.read((char*)&szBuf, sizeof(float));
      prefetch_label[item_id*fea_dim  + i ] = szBuf;
    }
    fin.close();

  }
}

INSTANTIATE_CLASS(TriRegDataLayer);
REGISTER_LAYER_CLASS(TriRegData);

}  // namespace caffe
