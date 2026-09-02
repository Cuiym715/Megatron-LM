from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.datasets.dspp_training import build_dspp_training_batch
from megatron.core.enums import ModelType
from megatron.core.kv_cache import dspp_packed_flash_attention
from megatron.core.pipeline_parallel import schedules


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class _TinyDsppLanguageModel(nn.Module):
    model_type = ModelType.encoder_or_decoder

    def __init__(self, vocab_size=47, hidden_size=32, heads=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def _qkv(self, hidden):
        shape = (*hidden.shape[:-1], self.heads, self.head_dim)
        return (
            self.q_proj(hidden).view(shape),
            self.k_proj(hidden).view(shape),
            self.v_proj(hidden).view(shape),
        )

    def forward(self, tokens, position_ids, attention_mask, kv_cache, labels):
        del position_ids, attention_mask
        hidden = self.embedding(tokens).transpose(0, 1)
        q, k, v = self._qkv(hidden)
        context = dspp_packed_flash_attention(
            q,
            k,
            v,
            state=kv_cache,
            meta=kv_cache.current_meta,
            layer_idx=0,
        )
        projected = self.out_proj(context.flatten(2)).transpose(0, 1)
        logits = self.lm_head(projected)
        return F.cross_entropy(
            logits.float().reshape(-1, self.vocab_size),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    def forward_unsliced(self, tokens, labels):
        hidden = self.embedding(tokens).unsqueeze(1)
        q, k, v = self._qkv(hidden)
        context = F.scaled_dot_product_attention(
            q.permute(1, 2, 0, 3),
            k.permute(1, 2, 0, 3),
            v.permute(1, 2, 0, 3),
            is_causal=True,
            dropout_p=0.0,
        ).permute(2, 0, 1, 3)
        projected = self.out_proj(context.flatten(2)).squeeze(1)
        logits = self.lm_head(projected)
        return F.cross_entropy(logits.float(), labels, reduction="none")


def _forward_step(batch, state, model):
    tokens, position_ids, attention_mask, labels = batch
    return model(
        tokens,
        position_ids,
        attention_mask,
        kv_cache=state,
        labels=labels,
    )


def _loss_func(mask, output):
    loss = (output.float() * mask).sum() / mask.sum()
    return loss, {"lm loss": loss.detach()}


def _patch_single_stage_runtime(monkeypatch, debug=1):
    monkeypatch.setattr(schedules, "cuda_sync_and_record", lambda **_: None)
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        schedules,
        "get_args",
        lambda: SimpleNamespace(
            variable_seq_debug_num_batches=debug,
            kaimm_cuda_synchronize_level=0,
            variable_seq_slicing=False,
            use_flash_attn=True,
        ),
    )


def test_dspp_schedule_matches_unsliced_loss_and_all_parameter_gradients(monkeypatch):
    _patch_single_stage_runtime(monkeypatch)
    raw_sequences = [
        torch.tensor(list(range(1, 20)), device="cuda"),
        torch.tensor(list(range(20, 29)), device="cuda"),
        torch.tensor([30, 31, 32, 33], device="cuda"),
        torch.tensor([34, 35, 36], device="cuda"),
    ]
    lengths = [sequence.numel() for sequence in raw_sequences]
    padded = torch.zeros(
        (len(raw_sequences), max(lengths)), dtype=torch.long, device="cuda"
    )
    for row, sequence in zip(padded, raw_sequences, strict=True):
        row[: sequence.numel()] = sequence
    training_batch = build_dspp_training_batch(
        padded,
        lengths,
        chunk_size=8,
        validate=True,
    )

    torch.manual_seed(901)
    actual_model = _TinyDsppLanguageModel().cuda().to(torch.bfloat16)
    reference_model = _TinyDsppLanguageModel().cuda().to(torch.bfloat16)
    reference_model.load_state_dict(actual_model.state_dict())
    data = iter(
        [
            (
                training_batch,
                partial(_loss_func, training_batch.loss_mask),
            )
        ]
    )

    losses = schedules.forward_backward_dspp_no_pipelining(
        forward_step_func=_forward_step,
        get_batch_func=lambda iterator: next(iterator),
        data_iterator=data,
        model=[actual_model],
        num_microbatches=1,
        micro_seq_length=8,
        kv_cache_class=None,
    )

    reference_losses = []
    for sequence in raw_sequences:
        reference_losses.append(
            reference_model.forward_unsliced(sequence[:-1], sequence[1:])
        )
    reference_loss = torch.cat(reference_losses).mean()
    reference_loss.backward()

    torch.testing.assert_close(
        losses[0]["lm loss"], reference_loss.detach(), rtol=2e-2, atol=2e-2
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        actual_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert actual.grad is not None, actual_name
        assert expected.grad is not None, expected_name
        torch.testing.assert_close(
            actual.grad.float(),
            expected.grad.float(),
            rtol=6e-2,
            atol=6e-2,
            msg=lambda message: f"{actual_name}: {message}",
        )


def test_dspp_schedule_forward_only_releases_retained_state(monkeypatch):
    _patch_single_stage_runtime(monkeypatch)
    tokens = torch.arange(1, 13, device="cuda").unsqueeze(0)
    training_batch = build_dspp_training_batch(
        tokens,
        [tokens.size(1)],
        chunk_size=4,
        validate=True,
    )
    model = _TinyDsppLanguageModel().cuda().to(torch.bfloat16)
    data = iter([(training_batch, partial(_loss_func, training_batch.loss_mask))])

    with torch.no_grad():
        result = schedules.forward_backward_dspp_no_pipelining(
            forward_step_func=_forward_step,
            get_batch_func=lambda iterator: next(iterator),
            data_iterator=data,
            model=model,
            num_microbatches=1,
            micro_seq_length=4,
            kv_cache_class=None,
            forward_only=True,
        )

    assert torch.isfinite(result[0]["lm loss"])


def test_dspp_schedule_weights_gradient_accumulation_by_valid_tokens(monkeypatch):
    _patch_single_stage_runtime(monkeypatch, debug=0)
    sequence_groups = [
        [torch.arange(1, 20, device="cuda"), torch.arange(20, 24, device="cuda")],
        [torch.arange(24, 33, device="cuda"), torch.arange(33, 36, device="cuda")],
    ]
    training_batches = []
    for sequences in sequence_groups:
        lengths = [sequence.numel() for sequence in sequences]
        padded = torch.zeros(
            (len(sequences), max(lengths)), dtype=torch.long, device="cuda"
        )
        for row, sequence in zip(padded, sequences, strict=True):
            row[: sequence.numel()] = sequence
        training_batches.append(
            build_dspp_training_batch(padded, lengths, chunk_size=8, validate=True)
        )

    torch.manual_seed(902)
    actual_model = _TinyDsppLanguageModel().cuda().to(torch.bfloat16)
    reference_model = _TinyDsppLanguageModel().cuda().to(torch.bfloat16)
    reference_model.load_state_dict(actual_model.state_dict())
    data = iter(
        [
            (training_batch, partial(_loss_func, training_batch.loss_mask))
            for training_batch in training_batches
        ]
    )
    losses = schedules.forward_backward_dspp_no_pipelining(
        forward_step_func=_forward_step,
        get_batch_func=lambda iterator: next(iterator),
        data_iterator=data,
        model=actual_model,
        num_microbatches=2,
        micro_seq_length=8,
        kv_cache_class=None,
    )

    reference_losses = [
        reference_model.forward_unsliced(sequence[:-1], sequence[1:])
        for sequences in sequence_groups
        for sequence in sequences
    ]
    reference_loss = torch.cat(reference_losses).mean()
    reference_loss.backward()

    logged_loss = sum(item["lm loss"] for item in losses) / len(losses)
    torch.testing.assert_close(
        logged_loss, reference_loss.detach(), rtol=2e-2, atol=2e-2
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        actual_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual.grad.float(),
            expected.grad.float(),
            rtol=6e-2,
            atol=6e-2,
            msg=lambda message: f"{actual_name}: {message}",
        )


def test_dspp_selector_selects_stage_local_v_for_pp3_vpp2(monkeypatch):
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 3,
    )
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_virtual_pipeline_model_parallel_world_size",
        lambda: 2,
    )

    assert (
        schedules.get_forward_backward_func(dspp=True)
        is schedules.pipelining_with_dspp_stage_local_v
    )


def test_dspp_v_padding_adds_only_the_missing_warmup_slots():
    assert schedules._pad_dspp_v_split_counts([6, 5], 3) == [6, 5]
    assert schedules._pad_dspp_v_split_counts([8, 3], 3) == [8, 4]
    padded = schedules._pad_dspp_v_split_counts([1, 1], 3)
    assert padded == [4, 4]
    schedules.build_slice_v_schedule(3, len(padded), padded)

    with pytest.raises(ValueError, match="at least two logical microbatches"):
        schedules._pad_dspp_v_split_counts([8], 3)
