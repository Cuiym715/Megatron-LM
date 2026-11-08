"""CP wrappers must find JIT extensions even when torch only returns a module."""
import sys
from types import ModuleType, SimpleNamespace

from megatron import fused_kernels


def test_returned_jit_extensions_are_importable(monkeypatch):
    modules = {}

    def load(*, name, **kwargs):
        modules[name] = ModuleType(name)
        return modules[name]

    for name in ('fast_flip_cuda', 'wrap_gemm_cuda', 'fast_cat_cuda'):
        # Preserve prior process state when the fixture rolls back mutations.
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(fused_kernels.cpp_extension, 'load', load)
    monkeypatch.setattr(fused_kernels.torch.cuda, 'get_device_capability', lambda: (8, 0))
    monkeypatch.setattr(fused_kernels, '_create_build_dir', lambda path: None)
    fused_kernels.load(SimpleNamespace(
        masked_softmax_fusion=False, use_fast_rope=False, rank=0,
        context_parallel_size=2, kaimm_offload_activation_ratio=0,
        kaimm_kv_cache_impl='chunked'))
    for name, module in modules.items():
        assert __import__(name) is module
