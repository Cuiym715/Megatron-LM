#include <torch/extension.h>
std::vector<at::Tensor> diagnostic_combined(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("combined", &diagnostic_combined); }
