#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Normalization.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <limits>
#include <tuple>


namespace at { namespace native {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<scalar_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a + b;
  }));
}

struct WelfordData {
  double mean;
  double m2;
  int64_t n;
  __host__ __device__ WelfordData() : mean(0), m2(0), n(0)  {}
  __device__ WelfordData(double mean, double m2, int64_t n) : mean(mean), m2(m2), n(n) {}
};

template <typename scalar_t, bool unbiased>
struct WelfordOps {
  static inline __device__ WelfordData reduce(const WelfordData& acc, scalar_t data) {
    double delta = data - acc.mean;
    double new_mean = acc.mean + delta / (acc.n + 1);
    double new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1
    };
  }
  static inline __device__ WelfordData combine(const WelfordData& a, const WelfordData& b) {
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
  static inline __device__ scalar_t project(const WelfordData& acc) {
    int64_t divisor = unbiased ? (acc.n - 1) : acc.n;
    return (divisor > 0) ? device_sqrt(acc.m2 / divisor) : NAN;
  }
  static inline __device__ WelfordData warp_shfl_down(const WelfordData& acc, int offset) {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
    };
  }
};


template <typename scalar_t, bool unbiased>
void std_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(iter, WelfordOps<scalar_t, unbiased> { }, WelfordData {});
}

#ifdef __HIPCC__
template <>
void sum_kernel_impl<int16_t, int16_t>(TensorIterator& iter) {
  // There is a Register Coalescing bug in LLVM causing the hcc
  // compiler segfaults:
  // https://bugs.llvm.org/show_bug.cgi?id=39602
  // To work around it, use int32 as the accumulate type.
  gpu_reduce_kernel<int16_t, int16_t>(iter, func_wrapper<int16_t> ([]GPU_LAMBDA(int32_t a, int32_t b) -> int32_t {
    return a + b;
  }));
}
#endif

template <typename scalar_t, typename acc_t=scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(iter, func_wrapper<scalar_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a * b;
  }), 1);
}

static void std_kernel_cuda(TensorIterator& iter, bool unbiased) {
  AT_DISPATCH_FLOATING_TYPES(iter.type(), "std", [&]() {
    if (unbiased) {
      std_kernel_impl<scalar_t, true>(iter);
    } else {
      std_kernel_impl<scalar_t, false>(iter);
    }
  });
}

static void sum_kernel_cuda(TensorIterator& iter) {
  if (iter.type().scalarType() == kHalf) {
    return sum_kernel_impl<at::Half, float>(iter);
  } else if (iter.type(1).scalarType() == kHalf && iter.type().scalarType() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::Half, float, float>(iter);
  }
  AT_DISPATCH_ALL_TYPES(iter.type(), "sum", [&]() {
    sum_kernel_impl<scalar_t>(iter);
  });
}

static void prod_kernel_cuda(TensorIterator& iter) {
  if (iter.type().scalarType() == kHalf) {
    return prod_kernel_impl<at::Half, float>(iter);
  }
  AT_DISPATCH_ALL_TYPES(iter.type(), "prod", [&]() {
    prod_kernel_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(std_stub, &std_kernel_cuda);
REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);

}} // namespace at::native
