#include <ATen/native/Welford.h>
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

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  gpu_reduce_kernel<scalar_t, scalar_t>(iter, WelfordOps<scalar_t> { unbiased, take_sqrt }, WelfordData {});
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

static void std_var_kernel_cuda(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.type(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
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

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);

}} // namespace at::native
