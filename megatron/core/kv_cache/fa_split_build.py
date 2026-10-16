"""Build the isolated stage-A extension from pinned FA2/CUTLASS sources.

No download occurs on import or normal kernel use. Fetch explicitly with
``python tools/build_dspp_fa_split.py --fetch``. The installed FA is untouched.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

NATIVE = Path(__file__).with_name("fa_split_src")
UPSTREAM = json.loads((NATIVE / "upstream.json").read_text())


def cache_root() -> Path:
    return Path(os.environ.get("DSPP_FA_SPLIT_CACHE", Path.home() / ".cache/dspp-fa-split")).expanduser().resolve()


def source_root() -> Path:
    return cache_root() / "flash-attention-2.7.4.post1"


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def fetch_sources() -> None:
    source = source_root()
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", UPSTREAM["flash_attention_tag"],
             UPSTREAM["flash_attention_repository"], str(source)], check=True,
        )
    if _git(source, "rev-parse", "HEAD") != UPSTREAM["flash_attention_commit"]:
        raise RuntimeError("FA source commit mismatch; refusing to change an existing checkout")
    subprocess.run(
        ["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1", "csrc/cutlass"],
        check=True,
    )


def validate_sources() -> None:
    source = source_root()
    if not (source / "csrc/cutlass/include/cute/tensor.hpp").is_file():
        raise RuntimeError("Pinned FA2/CUTLASS source missing; run tools/build_dspp_fa_split.py --fetch")
    for path, revision in ((source, UPSTREAM["flash_attention_commit"]),
                           (source / "csrc/cutlass", UPSTREAM["cutlass_commit"])):
        if _git(path, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"Source commit mismatch: {path}")
        subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--quiet"], check=True)
    header = source / "csrc/flash_attn/src/flash_bwd_kernel.h"
    if hashlib.sha256(header.read_bytes()).hexdigest() != UPSTREAM["header_sha256"]:
        raise RuntimeError("Upstream backward header checksum mismatch")


@functools.lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import load

    validate_sources()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build/test the stage-A extension")
    arch = torch.cuda.get_device_capability()
    # Match only the current target; do not compile every GPU in the host.
    if arch[0] not in (8, 9):
        raise RuntimeError("Stage A supports SM80-SM90 only")
    source = source_root()
    digest = hashlib.sha256()
    for name in ("backward.cu", "bindings.cpp", "flash_bwd_split.patch", "upstream.json"):
        digest.update((NATIVE / name).read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(f"{torch.__version__}:{torch.version.cuda}:{arch}:{torch._C._GLIBCXX_USE_CXX11_ABI}".encode())
    build = cache_root() / "build" / digest.hexdigest()[:16]
    build.mkdir(parents=True, exist_ok=True)
    generated = build / "dspp_flash_bwd_kernel.h"
    # Produce the patched header separately, never editing upstream/installed FA.
    with tempfile.TemporaryDirectory(dir=build) as tmp:
        patched = Path(tmp) / generated.name
        subprocess.run(
            ["patch", "--batch", "--silent", "--fuzz=0", "--output", str(patched),
             str(source / "csrc/flash_attn/src/flash_bwd_kernel.h"),
             str(NATIVE / "flash_bwd_split.patch")], check=True,
        )
        if not generated.exists() or generated.read_bytes() != patched.read_bytes():
            patched.replace(generated)
    return load(
        name="dspp_fa_split_cuda",
        sources=[str(NATIVE / "bindings.cpp"), str(NATIVE / "backward.cu")],
        extra_include_paths=[str(build), str(source / "csrc/flash_attn/src"),
                             str(source / "csrc/cutlass/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                          "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
                          f"-DDSPP_FA_SPLIT_ARCH={arch[0]}{arch[1]}",
                          f"-gencode=arch=compute_{arch[0]}{arch[1]},code=sm_{arch[0]}{arch[1]}"],
        build_directory=str(build),
        verbose=os.environ.get("DSPP_FA_SPLIT_VERBOSE", "0") == "1",
    )
