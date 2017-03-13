# Introduction
本项目是基于MPI和NCCL实现的多机多卡并行Caffe。针对人脸识别的特定问题，进行了训练速度和显存占用方面的若干优化，同时保有执行其他任务的通用性

# Requirements
- [nvidia driver](http://www.nvidia.com/Download/index.aspx) = 367.57
  * 仅在这个driver上通过稳定性测试，更新或更老的driver未充分测试
  * 部分driver如375.20, 将会不定期crash
  * 如有新发现，请报告在[issue#3](http://192.168.50.80/jianyi/caffe-mpi/issues/3)

- [cuda](https://developer.nvidia.com/cuda-downloads) >= 8.0

- [openmpi](https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.0.tar.gz) >= 2.0.0
  * $ ./configure --prefix=/usr/local/mpi --with-cuda
  * make && make install

- [nccl](https://github.com/NVIDIA/nccl)
  * $ make CUDA_HOME=cuda_install_path all -j
  * $ make PREFIX=/usr/local/nccl install
  * 在LD_LIBRARY_PATH中添加/usr/local/nccl/lib 

# Get Started
- 单机单卡
  * $ mkdir build && cd build
  * $ cmake .. -DUSE_MPI=OFF [-DCMAKE_BUILD_TYPE=Debug]
  * $ make all -j
  * $ cd ..
  * $ ./build/tools/caffe train ... as usual

- 单机多卡（以4卡为例）
  * $ mkdir build && cd build
  * $ cmake .. -DUSE_MPI=ON [-DCMAKE_BUILD_TYPE=Debug]
  * $ make all -j
  * $ cd ..
  * 编辑你的solver prototxt, 增加device_id: [0,1,2,3]  
  * $ mpirun -np 4 ./build/tools/caffe train ...

- 多机多卡
  * 由于用了nccl，目前不支持多机多卡。但未来计划结合MPI和NCCL，使之具备跨机能力。敬请期待

- 高级用法
  * [wiki usage](http://192.168.50.80/jianyi/caffe-mpi/wikis/home)

# For Developer
- ChangeLog
  * 支持NCCL通讯。通讯效率大幅提升65%-100%，同时也失去了跨机能力 - [benchmark](http://192.168.50.80/jianyi/caffe-mpi/wikis/%5Bbenchmark%5D%E5%B9%B6%E8%A1%8C%E8%AE%AD%E7%BB%83%E9%80%9F%E5%BA%A6)
  * 支持批量同步gradients。通讯效率提升16% - [benchmark](http://192.168.50.80/jianyi/caffe-mpi/wikis/%5Bbenchmark%5D%E5%B9%B6%E8%A1%8C%E8%AE%AD%E7%BB%83%E9%80%9F%E5%BA%A6) - [usage](http://192.168.50.80/jianyi/caffe-mpi/wikis/%E9%80%9A%E8%AE%AF%E6%A8%A1%E5%BC%8F%E5%88%87%E6%8D%A2)
  * 支持fc和softmax loss融合。其显存占用和data层的batch size解耦，同时显存锐减2/3 - [benchmark](http://192.168.50.80/jianyi/caffe-mpi/wikis/%5Bbenchmark%5D%E5%B9%B6%E8%A1%8C%E8%AE%AD%E7%BB%83%E6%98%BE%E5%AD%98) - [usage](http://192.168.50.80/jianyi/caffe-mpi/wikis/%E8%9E%8D%E5%90%88%E5%B1%82%E7%94%A8%E6%B3%95)
  * 支持CUDA统一寻址 - [benchmark]() - [usage](http://192.168.50.80/jianyi/caffe-mpi/wikis/%E5%BC%80%E5%90%AF%E7%BB%9F%E4%B8%80%E5%AF%BB%E5%9D%80)
  * 支持fc + softmax loss的模型并行。获得线性加速比 - [benchmark]() - [usage]()
  * 支持"dry run"显存优化。显存锐减30%~50% - [benchmark](http://192.168.50.80/jianyi/caffe-mpi/wikis/%5Bbenchmark%5D%E5%B9%B6%E8%A1%8C%E8%AE%AD%E7%BB%83%E6%98%BE%E5%AD%98) - [usage](http://192.168.50.80/jianyi/caffe-mpi/wikis/%E5%BC%80%E5%90%AF%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96)
  * 支持kd模型压缩。dense/sparse soft target & 高效并行压缩 - [usage1](http://192.168.50.80/jianyi/caffe-mpi/wikis/kd%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9) - [usage2](http://192.168.50.80/jianyi/caffe-mpi/wikis/%E4%BA%BA%E8%84%B8%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9)

- Milestone
  * nccl + mpi的通讯组件。用mpi处理跨机通讯，用nccl处理本机通讯 - @huangdian - [Issue#2](http://192.168.50.80/jianyi/caffe-mpi/issues/2)
  * LightRNN思路改造fc + softmax loss - @Medivhna
  * LVQ思路改进triplet loss - @xiong

- Contribute
  * 贡献本项目的基本步骤:
  * clone master: git clone http://192.168.50.80/jianyi/caffe-mpi.git
  * cd caffe-mpi
  * 切换到feature分支: git checkout -b brach-name # brach-names随意取，与feature相关
  * 修改代码
  * git add .
  * git commit -m"你的注释"
  * git push origin brach-name
  * 此时进入待合并状态 
