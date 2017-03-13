#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer() : initted_(false), running_(false), 
            has_run_at_least_once_(false) {}
  virtual ~Timer() {};
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual float MilliSeconds() = 0;
  virtual float MicroSeconds() = 0;
  virtual float Seconds() {
    return MilliSeconds() / 1000.;
  }
  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

#ifndef CPU_ONLY
class GPUTimer : public Timer {
 public:
  explicit GPUTimer();
  virtual ~GPUTimer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();

 protected:
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
};
#endif

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();

 protected:
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
