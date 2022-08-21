#include <torch/extension.h>

#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> inser_points(at::Tensor x);

std::tuple<at::Tensor, at::Tensor> inser_points(at::Tensor x) {
  CHECK_INPUT(x);
  return insert_vox_points(x);
}
