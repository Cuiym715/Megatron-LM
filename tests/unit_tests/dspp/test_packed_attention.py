import gc

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache.dspp_packed_attention import (
    DsppSequenceKVState,
    dspp_packed_flash_attention,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _make_qkv(lengths, *, seed, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    result = []
    for length in lengths:
        tensors = tuple(
            torch.randn(
                length,
                2,
                16,
                device="cuda",
                dtype=dtype,
                generator=generator,
                requires_grad=True,
            )
            for _ in range(3)
        )
        result.append(tensors)
    return result


def _pad_physical_qkv(source, meta):
    result = []
    for tensor_index in range(3):
        valid = torch.cat(
            [
                source[item.sequence_id][tensor_index][
                    item.token_offset : item.token_offset + item.token_length
                ]
                for item in meta.items
            ],
            dim=0,
        )
        padding = valid.new_zeros(meta.chunk_size - valid.size(0), *valid.shape[1:])
        result.append(torch.cat((valid, padding), dim=0).unsqueeze(1))
    return result


def _run_segmented(plan, source, output_grads=None):
    state = DsppSequenceKVState(validate_runtime=True)
    outputs = []
    for meta in plan.microbatches:
        q, k, v = _pad_physical_qkv(source, meta)
        outputs.append(
            dspp_packed_flash_attention(
                q,
                k,
                v,
                state=state,
                meta=meta,
                layer_idx=7,
            )
        )
    if output_grads is not None:
        for meta, output in reversed(
            list(zip(plan.microbatches, outputs, strict=True))
        ):
            physical_grad = output.new_zeros(output.shape)
            cursor = 0
            for item in meta.items:
                end = cursor + item.token_length
                physical_grad[cursor:end, 0] = output_grads[item.sequence_id][
                    item.token_offset : item.token_offset + item.token_length
                ]
                cursor = end
            torch.autograd.backward(output, physical_grad)
        state.finish_iteration()
    return outputs, state


def _run_unsliced(source, output_grads):
    outputs = []
    for (q, k, v), grad in zip(source, output_grads, strict=True):
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=True,
            dropout_p=0.0,
        ).transpose(0, 1)
        torch.autograd.backward(output, grad)
        outputs.append(output)
    return outputs


def _unpack_outputs(plan, outputs):
    unpacked = {sequence_id: [] for sequence_id in range(len(plan.sequence_lengths))}
    for meta, output in zip(plan.microbatches, outputs, strict=True):
        cursor = 0
        for item in meta.items:
            end = cursor + item.token_length
            unpacked[item.sequence_id].append(
                (item.segment_id, output[cursor:end, 0].detach())
            )
            cursor = end
    return [
        torch.cat([value for _, value in sorted(parts)], dim=0)
        for parts in unpacked.values()
    ]


def test_packed_flash_matches_unsliced_outputs_and_all_qkv_gradients():
    lengths = [18, 8, 3, 2]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    segmented_source = _make_qkv(lengths, seed=123)
    reference_source = [
        tuple(t.detach().clone().requires_grad_() for t in qkv)
        for qkv in segmented_source
    ]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(456)
    output_grads = [
        torch.randn(
            length, 2, 16, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for length in lengths
    ]

    segmented_outputs, state = _run_segmented(plan, segmented_source, output_grads)
    reference_outputs = _run_unsliced(reference_source, output_grads)
    unpacked_outputs = _unpack_outputs(plan, segmented_outputs)

    state.finish_iteration()
    for actual, expected in zip(unpacked_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
    for actual_qkv, expected_qkv in zip(
        segmented_source, reference_source, strict=True
    ):
        for actual, expected in zip(actual_qkv, expected_qkv, strict=True):
            torch.testing.assert_close(
                actual.grad.float(), expected.grad.float(), rtol=3e-2, atol=3e-2
            )


def test_packed_sequences_are_attention_isolated():
    lengths = [10, 3, 2]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    first_source = _make_qkv(lengths, seed=100)
    second_source = [
        tuple(t.detach().clone().requires_grad_() for t in qkv) for qkv in first_source
    ]
    for sequence_id in (1, 2):
        for tensor in second_source[sequence_id]:
            with torch.no_grad():
                tensor.add_(100)

    first_outputs, _ = _run_segmented(plan, first_source)
    second_outputs, _ = _run_segmented(plan, second_source)
    first_unpacked = _unpack_outputs(plan, first_outputs)
    second_unpacked = _unpack_outputs(plan, second_outputs)

    torch.testing.assert_close(first_unpacked[0], second_unpacked[0], rtol=0, atol=0)
    assert not torch.equal(first_unpacked[1], second_unpacked[1])


def test_reverse_backward_drains_state_for_100_iterations():
    lengths = [10, 3, 2]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    for iteration in range(100):
        source = _make_qkv(lengths, seed=iteration)
        output_grads = [torch.ones_like(qkv[0]) for qkv in source]
        _, state = _run_segmented(plan, source, output_grads)
        state.finish_iteration()
    del source, output_grads, state
    gc.collect()
    torch.cuda.synchronize()
    final_allocated = torch.cuda.memory_allocated()

    assert final_allocated - baseline < 8 * 1024 * 1024


def test_forward_order_violation_is_rejected():
    plan = build_dspp_batch_plan([18], chunk_size=8)
    source = _make_qkv([18], seed=1)
    state = DsppSequenceKVState(validate_runtime=True)
    q, k, v = _pad_physical_qkv(source, plan.microbatches[1])

    with pytest.raises(RuntimeError, match="missing KV history"):
        dspp_packed_flash_attention(
            q,
            k,
            v,
            state=state,
            meta=plan.microbatches[1],
            layer_idx=0,
        )


class _TinyAttentionBlock(nn.Module):
    def __init__(self, hidden_size=32, heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def _qkv(self, hidden):
        shape = (*hidden.shape[:-1], self.heads, self.head_dim)
        return (
            self.q_proj(hidden).view(shape),
            self.k_proj(hidden).view(shape),
            self.v_proj(hidden).view(shape),
        )

    def forward_unsliced(self, hidden):
        q, k, v = self._qkv(hidden)
        context = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=True,
            dropout_p=0.0,
        ).transpose(0, 1)
        return self.out_proj(context.flatten(1))

    def forward_packed(self, hidden, state, meta):
        q, k, v = self._qkv(hidden)
        context = dspp_packed_flash_attention(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            state=state,
            meta=meta,
            layer_idx=0,
        )[:, 0]
        return self.out_proj(context.flatten(1))


def test_parameterized_block_matches_unsliced_input_and_parameter_gradients():
    lengths = [18, 16, 3, 2]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    torch.manual_seed(2026)
    segmented_model = _TinyAttentionBlock().cuda().to(torch.bfloat16)
    reference_model = _TinyAttentionBlock().cuda().to(torch.bfloat16)
    reference_model.load_state_dict(segmented_model.state_dict())

    segmented_hidden = [
        torch.randn(length, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for length in lengths
    ]
    reference_hidden = [
        hidden.detach().clone().requires_grad_() for hidden in segmented_hidden
    ]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2027)
    output_grads = [
        torch.randn(
            length, 32, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for length in lengths
    ]

    state = DsppSequenceKVState(validate_runtime=True)
    segmented_outputs = []
    for meta in plan.microbatches:
        valid = torch.cat(
            [
                segmented_hidden[item.sequence_id][
                    item.token_offset : item.token_offset + item.token_length
                ]
                for item in meta.items
            ]
        )
        padded = torch.cat(
            (valid, valid.new_zeros(meta.chunk_size - valid.size(0), valid.size(1)))
        )
        segmented_outputs.append(segmented_model.forward_packed(padded, state, meta))

    for meta, output in reversed(
        list(zip(plan.microbatches, segmented_outputs, strict=True))
    ):
        physical_grad = output.new_zeros(output.shape)
        cursor = 0
        for item in meta.items:
            end = cursor + item.token_length
            physical_grad[cursor:end] = output_grads[item.sequence_id][
                item.token_offset : item.token_offset + item.token_length
            ]
            cursor = end
        torch.autograd.backward(output, physical_grad)
    state.finish_iteration()

    for hidden, output_grad in zip(reference_hidden, output_grads, strict=True):
        output = reference_model.forward_unsliced(hidden)
        torch.autograd.backward(output, output_grad)

    for actual, expected in zip(segmented_hidden, reference_hidden, strict=True):
        torch.testing.assert_close(
            actual.grad.float(), expected.grad.float(), rtol=5e-2, atol=5e-2
        )
    for (actual_name, actual), (expected_name, expected) in zip(
        segmented_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual.grad.float(),
            expected.grad.float(),
            rtol=6e-2,
            atol=6e-2,
            msg=lambda message: f"parameter {actual_name}: {message}",
        )


class _TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.block = _TinyAttentionBlock(hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_unsliced(self, tokens):
        return self.output(self.block.forward_unsliced(self.embedding(tokens)))

    def forward_packed(self, tokens, state, meta):
        return self.output(
            self.block.forward_packed(self.embedding(tokens), state, meta)
        )


def test_materialized_language_model_loss_and_every_parameter_gradient_match():
    lengths = [18, 16, 3, 2]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    sequences = [
        torch.tensor([(sequence_id * 13 + index) % 64 for index in range(length + 1)])
        for sequence_id, length in enumerate(lengths)
    ]
    batches = plan.materialize(sequences, pad_token_id=0)
    torch.manual_seed(3030)
    segmented_model = _TinyLanguageModel().cuda().to(torch.bfloat16)
    reference_model = _TinyLanguageModel().cuda().to(torch.bfloat16)
    reference_model.load_state_dict(segmented_model.state_dict())
    total_tokens = sum(lengths)

    state = DsppSequenceKVState(validate_runtime=True)
    segmented_losses = []
    for batch in batches:
        tokens = batch.tokens[0].cuda()
        labels = batch.labels[0, : batch.meta.valid_token_count].cuda()
        logits = segmented_model.forward_packed(tokens, state, batch.meta)
        segmented_losses.append(
            F.cross_entropy(
                logits[: batch.meta.valid_token_count].float(),
                labels,
                reduction="sum",
            )
            / total_tokens
        )
    for loss in reversed(segmented_losses):
        loss.backward()
    state.finish_iteration()

    reference_losses = []
    for sequence in sequences:
        tokens = sequence[:-1].cuda()
        labels = sequence[1:].cuda()
        logits = reference_model.forward_unsliced(tokens)
        loss = F.cross_entropy(logits.float(), labels, reduction="sum") / total_tokens
        reference_losses.append(loss)
        loss.backward()

    torch.testing.assert_close(
        torch.stack([loss.detach() for loss in segmented_losses]).sum(),
        torch.stack([loss.detach() for loss in reference_losses]).sum(),
        rtol=2e-3,
        atol=2e-3,
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        segmented_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual.grad.float(),
            expected.grad.float(),
            rtol=7e-2,
            atol=7e-2,
            msg=lambda message: f"parameter {actual_name}: {message}",
        )


def test_grouped_query_attention_matches_unsliced_gradients():
    lengths = [10, 3]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(4040)
    segmented = [
        (
            torch.randn(
                length,
                4,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            ),
            torch.randn(
                length,
                2,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            ),
            torch.randn(
                length,
                2,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            ),
        )
        for length in lengths
    ]
    reference = [
        tuple(tensor.detach().clone().requires_grad_() for tensor in qkv)
        for qkv in segmented
    ]
    grads = [torch.randn_like(qkv[0]) for qkv in segmented]

    state = DsppSequenceKVState(validate_runtime=True)
    outputs = []
    for meta in plan.microbatches:
        physical = []
        for tensor_index in range(3):
            valid = torch.cat(
                [
                    segmented[item.sequence_id][tensor_index][
                        item.token_offset : item.token_offset + item.token_length
                    ]
                    for item in meta.items
                ]
            )
            physical.append(
                torch.cat(
                    (
                        valid,
                        valid.new_zeros(
                            meta.chunk_size - valid.size(0), *valid.shape[1:]
                        ),
                    ),
                    dim=0,
                ).unsqueeze(1)
            )
        outputs.append(
            dspp_packed_flash_attention(
                *physical,
                state=state,
                meta=meta,
                layer_idx=0,
            )
        )
    for meta, output in reversed(list(zip(plan.microbatches, outputs, strict=True))):
        physical_grad = output.new_zeros(output.shape)
        cursor = 0
        for item in meta.items:
            end = cursor + item.token_length
            physical_grad[cursor:end, 0] = grads[item.sequence_id][
                item.token_offset : item.token_offset + item.token_length
            ]
            cursor = end
        torch.autograd.backward(output, physical_grad)
    state.finish_iteration()

    for (q, k, v), grad in zip(reference, grads, strict=True):
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=True,
            dropout_p=0.0,
            enable_gqa=True,
        ).transpose(0, 1)
        torch.autograd.backward(output, grad)

    for actual_qkv, expected_qkv in zip(segmented, reference, strict=True):
        for actual, expected in zip(actual_qkv, expected_qkv, strict=True):
            torch.testing.assert_close(
                actual.grad.float(), expected.grad.float(), rtol=4e-2, atol=4e-2
            )
