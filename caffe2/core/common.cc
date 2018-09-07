#include <atomic>

#include "caffe2/core/common.h"

#include "common/fb303/cpp/FacebookBase2.h"

namespace caffe2 {

CounterThenLog::CounterThenLog(const char* name) : name_(name), count_(0) {
  facebook::fbData->addStatExportType(name_, facebook::stats::SUM);
}
void CounterThenLog::bump(int i) {
  count_ += 1;
  facebook::fbData->addStatValue(name_);
  if (count_ % 1000000 < i) {
    std::cerr << "CounterThenLog: " << name_ << " = " << count_;
  }
}

// A global variable to mark if Caffe2 has cuda linked to the current runtime.
// Do not directly use this variable, but instead use the HasCudaRuntime()
// function below.
std::atomic<bool> g_caffe2_has_cuda_linked{false};
std::atomic<bool> g_caffe2_has_hip_linked{false};

bool HasCudaRuntime() {
  return g_caffe2_has_cuda_linked.load();
}

bool HasHipRuntime() {
  return g_caffe2_has_hip_linked.load();
}

namespace internal {
void SetCudaRuntimeFlag() {
  g_caffe2_has_cuda_linked.store(true);
}

void SetHipRuntimeFlag() {
  g_caffe2_has_hip_linked.store(true);
}
} // namespace internal

const std::map<string, string>& GetBuildOptions() {
#ifndef CAFFE2_BUILD_STRINGS
#define CAFFE2_BUILD_STRINGS {}
#endif
  static const std::map<string, string> kMap = CAFFE2_BUILD_STRINGS;
  return kMap;
}

} // namespace caffe2
