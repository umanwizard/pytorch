#include <c10/macros/Macros.h>
#if (defined(__CUDACC__) || defined(__HIPCC__))
#include <THC/THCDeviceUtils.cuh>
#include <ATen/native/cuda/Normalization.cuh>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif

namespace at { namespace native {

struct WelfordData {
  double mean;
  double m2;
  int64_t n;
  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0)  {}
  C10_DEVICE WelfordData(double mean, double m2, int64_t n) : mean(mean), m2(m2), n(n) {}
};


template <typename scalar_t>
struct WelfordOps {
  bool unbiased;
  bool take_sqrt;
 public:
  inline C10_DEVICE WelfordData reduce(WelfordData acc, scalar_t data) const {
    double delta = data - acc.mean;
    double new_mean = acc.mean + delta / (acc.n + 1);
    double new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1
    };
  }
  inline C10_DEVICE WelfordData combine(WelfordData a, WelfordData b) const {
    if (a.n == 0) {
      return b;
    }
    if (b.n == 0) {
      return a;
    }
    double delta = b.mean - a.mean;
    int64_t new_count = a.n + b.n;
    double nb_over_n = (double)b.n / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.n * nb_over_n,
      new_count
    };
  }
  inline C10_DEVICE scalar_t project(WelfordData acc) const {
    int64_t divisor = unbiased ? (acc.n - 1) : acc.n;
    return (divisor > 0) ? 
      (take_sqrt ? device_sqrt(acc.m2 / divisor) : (acc.m2 / divisor))
      : NAN;
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ WelfordData warp_shfl_down(WelfordData acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
    };
  }
#endif
  WelfordOps(bool unbiased, bool take_sqrt)
    : unbiased(unbiased), take_sqrt(take_sqrt) {
  }
};

}} // namespace at::native
