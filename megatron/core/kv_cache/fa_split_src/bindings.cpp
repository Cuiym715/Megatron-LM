// Experimental FA2 backward split. Production FlashAttention is not replaced.
#include <torch/extension.h>

at::Tensor split_prepare(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor split_dq(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, double, bool);
std::vector<at::Tensor> split_dkv(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, double, int64_t, int64_t, bool);
std::vector<at::Tensor> split_dq_current(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                       const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                       const at::Tensor&, double, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &split_prepare);
    m.def("dq", &split_dq,
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("o"), pybind11::arg("dout"), pybind11::arg("lse"),
          pybind11::arg("delta"), pybind11::arg("scale"), pybind11::arg("deterministic") = false);
    m.def("dq_current", &split_dq_current,
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("o"), pybind11::arg("dout"), pybind11::arg("lse"),
          pybind11::arg("delta"), pybind11::arg("scale"), pybind11::arg("deterministic") = false);
    m.def("dkv", &split_dkv,
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("o"), pybind11::arg("dout"), pybind11::arg("lse"),
          pybind11::arg("delta"), pybind11::arg("scale"),
          pybind11::arg("begin"), pybind11::arg("end"), pybind11::arg("compact") = false);
}
