# Source provenance

The extension uses FlashAttention v2.7.4.post1, commit
`5231d95fe13733fb534c01895f7ea88c6a6c7793`, with its CUTLASS submodule at
`c506e16788cb08416a4a57e11a9067beeee29420`. See `upstream.json` for the upstream
backward-header checksum and repository URL. Full license notices are retained
in `LICENSE.flash-attention` and `LICENSE.cutlass`.

`flash_bwd_split.patch` modifies Tri Dao's `flash_bwd_kernel.h` by adding
compile-time dQ/dKV switches to the existing column-block backward routine.
It suppresses the unwanted gradient GEMMs and writes, including the early-exit
zero writes, while retaining common probability/dS calculations and barriers.
The custom launcher originally instantiated only the independent variants, using the
upstream head-dim-128 BF16 tile configuration for L40. Shared preprocessing and
dQ conversion use unchanged upstream routines. The installed FA library is not
patched or replaced. No performance claim is made for these initial variants.

Sources are fetched only on explicit request, checked against pinned commits,
and patched into the build cache. `backward.cu` supplies isolated parameter setup,
validation and launchers; `bindings.cpp` exposes the separate operations.

Stage B extends the launcher (without changing the upstream-header patch) to
unequal Q/K lengths, native GQA head mapping, and selected KV tile ranges.
It follows the upstream `flash_api.cpp` backward convention: compact K/V inputs,
BF16 partial dK/dV per Q head, then ATen group summation into compact KV-head
outputs (FP32 accumulation for BF16 sum). Boundary-tile surplus is excluded
from output. The Python wrapper dispatches independent varlen sequences using
their full-context O/LSE; it does not replace FA2 with a dense implementation.

The combined-current entry additionally instantiates the unchanged full-gradient
column-block routine for tiles intersecting current KV, and dQ-only for tiles
wholly within history. It reads only current rows from the per-Q-head dKV
workspace; a nonaligned boundary tile's surplus history output is discarded.
The original upstream-header patch, tile configuration, GQA reduction and dQ
conversion remain unchanged. Independent historical dKV is still computed later.

The optional diagnostic `deterministic=True` dQ entry follows pinned FA2's
seq-K parallel deterministic launcher: ceil(SM_count / Q_heads) independent
FP32 dQ partitions (batch=1), a fixed strided traversal of KV tiles per CTA,
and the unchanged upstream `convert_dQ(params, nsplits)` reduction. The
combined-current entry keeps the same dQ-only/current-dKV tile selection.
No upstream header patch is added for this option. It costs extra temporary
FP32 storage; the Python/native defaults remain false, and production
attention does not enable it. It is for explicit numerical diagnosis only.
