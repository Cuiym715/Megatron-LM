# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Check the availability of apex
    assert multi_tensor_applier is not None and amp_C is not None, \
        "apex is not available, please install it from https://github.com/NVIDIA/apex"

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    args = get_args()

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    if not args.use_flash_attn:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    if args.use_rotary_position_embeddings:
        position_ids = 0    # the start pos.
        assert not (reset_position_ids or reset_attention_mask), 'RoPE has no position ids or attention mask to reset.'
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    if args.use_flash_attn:
        attention_mask = None
    else:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_sliced_batch(tokens, position_ids, attention_mask, labels, micro_seq_length=None):
    if not micro_seq_length:
        return [(tokens, position_ids, attention_mask, labels)]

    tokens = tokens.split(micro_seq_length, dim=1)
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.split(micro_seq_length, dim=-1)
    else:
        position_ids = [i * micro_seq_length for i in range(len(tokens))]
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.split(micro_seq_length, dim=-1)
    else:
        attention_mask = [None] * len(tokens)
    labels = labels.split(micro_seq_length, dim=1)
    return list(zip(tokens, position_ids, attention_mask, labels))


def get_variable_sliced_batch(tokens, position_ids, attention_mask, labels,
                              loss_mask, micro_seq_length, pad_token_id=-1,
                              pad_chunks_to_multiple=1):
    """Slice a right-padded batch into fixed-size chunks.

    The original SlimPipe splitter assumes every microbatch has
    ``seq_length / micro_seq_length`` chunks. This variant trims a batch to the
    smallest chunk-aligned length that covers its non-padding tokens, then
    returns only those chunks.
    """
    assert micro_seq_length, "variable sequence slicing requires micro_seq_length > 0"
    args = get_args()
    seq_length = tokens.size(1)
    lengths = None
    if pad_token_id >= 0:
        valid_tokens = tokens.ne(pad_token_id)
        lengths = valid_tokens.long().sum(dim=1)
        # Labels are shifted left by one token; keep one position if the sample
        # is all padding so downstream tensor shapes remain valid.
        max_len = int(lengths.max().item()) if lengths.numel() else seq_length
        max_len = max(max_len, 1)
        loss_mask = loss_mask.masked_fill(labels.eq(pad_token_id), 0.0)
    else:
        max_len = seq_length

    num_chunks = (max_len + micro_seq_length - 1) // micro_seq_length
    unpadded_num_chunks = num_chunks
    if pad_chunks_to_multiple > 1:
        num_chunks = ((num_chunks + pad_chunks_to_multiple - 1) //
                      pad_chunks_to_multiple) * pad_chunks_to_multiple
    padded_len = num_chunks * micro_seq_length

    # DEBUG for variable length training.
    debug_limit = getattr(args, 'variable_seq_debug_num_batches', 0)
    debug_count = getattr(get_variable_sliced_batch, '_debug_count', 0)
    if debug_limit > debug_count:
        if lengths is not None:
            lengths_cpu = lengths.detach().cpu().tolist()
            length_summary = (
                f"valid_lengths={lengths_cpu}, min={min(lengths_cpu)}, "
                f"max={max(lengths_cpu)}"
            )
        else:
            length_summary = "valid_lengths=all tokens treated as valid"
        print_rank_0(
            "[variable-seq][split] "
            f"microbatch={debug_count}, batch={tokens.size(0)}, "
            f"input_seq_len={seq_length}, {length_summary}, "
            f"chunk_size={micro_seq_length}, chunks={unpadded_num_chunks}, "
            f"padded_chunks={num_chunks}, padded_seq_len={padded_len}, "
            f"pad_chunks_to_multiple={pad_chunks_to_multiple}"
        )
        get_variable_sliced_batch._debug_count = debug_count + 1

    def pad_or_trim_2d(x, pad_value):
        if x.size(1) >= padded_len:
            return x[:, :padded_len].contiguous()
        pad_width = padded_len - x.size(1)
        return torch.nn.functional.pad(x, (0, pad_width), value=pad_value).contiguous()

    pad_value = pad_token_id if pad_token_id >= 0 else 0
    tokens = pad_or_trim_2d(tokens, pad_value)
    labels = pad_or_trim_2d(labels, pad_value)
    loss_mask = pad_or_trim_2d(loss_mask, 0.0)
    if isinstance(position_ids, torch.Tensor):
        if position_ids.size(1) >= padded_len:
            position_ids = position_ids[:, :padded_len].contiguous()
        else:
            extra = torch.arange(position_ids.size(1), padded_len,
                                 dtype=position_ids.dtype,
                                 device=position_ids.device)
            extra = extra.unsqueeze(0).expand(position_ids.size(0), -1)
            position_ids = torch.cat((position_ids, extra), dim=1).contiguous()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask[..., :padded_len].contiguous()

    slices = get_sliced_batch(tokens, position_ids, attention_mask, labels,
                              micro_seq_length)
    assert len(slices) == num_chunks
    return slices, loss_mask


def get_packed_variable_sliced_batch(tokens_with_labels, micro_seq_length,
                                     seq_length, pad_token_id=-1,
                                     pad_chunks_to_multiple=1):
    """Pack document-level variable-length samples into chunk-sized slices.

    ``tokens_with_labels`` is a padded CPU/GPU tensor shaped ``[batch, length]``.
    Each row is treated as one independent sequence and truncated to
    ``seq_length + 1`` tokens. The function packs sequence fragments into
    fixed-size chunks of ``micro_seq_length`` model tokens. Attention is causal
    within each original sequence and masked across different packed sequences.
    """
    assert micro_seq_length > 0, "variable sequence packing requires micro_seq_length > 0"
    args = get_args()
    device = tokens_with_labels.device
    max_input_len = seq_length + 1
    pad_value = pad_token_id if pad_token_id >= 0 else 0

    chunks = []
    loss_masks = []
    history_seq_ids = []
    history_positions = []
    cur_tokens = []
    cur_labels = []
    cur_loss_mask = []
    cur_position_ids = []
    cur_seq_ids = []
    cur_positions = []
    seq_lengths = []
    chunk_segments = []
    next_seq_id = 0

    def flush_chunk():
        nonlocal cur_tokens, cur_labels, cur_loss_mask, cur_position_ids
        nonlocal cur_seq_ids, cur_positions
        if not cur_tokens:
            return
        real_len = len(cur_tokens)
        pad_len = micro_seq_length - real_len
        token_values = cur_tokens + [pad_value] * pad_len
        label_values = cur_labels + [pad_value] * pad_len
        loss_values = cur_loss_mask + [0.0] * pad_len
        position_values = cur_position_ids + [0] * pad_len
        seq_id_values = cur_seq_ids + [-1] * pad_len
        position_key_values = cur_positions + [-1] * pad_len

        key_seq_ids = history_seq_ids + seq_id_values
        key_positions = history_positions + position_key_values
        query_seq_ids = torch.tensor(seq_id_values, dtype=torch.long,
                                     device=device).unsqueeze(1)
        query_positions = torch.tensor(position_key_values, dtype=torch.long,
                                       device=device).unsqueeze(1)
        key_seq_ids_tensor = torch.tensor(key_seq_ids, dtype=torch.long,
                                          device=device).unsqueeze(0)
        key_positions_tensor = torch.tensor(key_positions, dtype=torch.long,
                                            device=device).unsqueeze(0)
        allowed = (
            (query_seq_ids >= 0) &
            (query_seq_ids == key_seq_ids_tensor) &
            (key_positions_tensor >= 0) &
            (key_positions_tensor <= query_positions)
        )
        mask = ~allowed

        token_tensor = torch.tensor(token_values, dtype=torch.long,
                                   device=device).unsqueeze(0)
        label_tensor = torch.tensor(label_values, dtype=torch.long,
                                   device=device).unsqueeze(0)
        position_tensor = torch.tensor(position_values, dtype=torch.long,
                                      device=device).unsqueeze(0)
        loss_tensor = torch.tensor(loss_values, dtype=torch.float,
                                  device=device).unsqueeze(0)
        attention_mask = mask.view(1, 1, micro_seq_length, len(key_seq_ids))
        chunks.append((token_tensor, position_tensor, attention_mask, label_tensor))
        loss_masks.append(loss_tensor)

        history_seq_ids.extend(seq_id_values)
        history_positions.extend(position_key_values)
        chunk_segments.append([
            (seq_id, start, length)
            for seq_id, start, length in _summarize_packed_segments(seq_id_values, position_key_values)
        ])
        cur_tokens = []
        cur_labels = []
        cur_loss_mask = []
        cur_position_ids = []
        cur_seq_ids = []
        cur_positions = []

    for row in tokens_with_labels:
        row = row[:max_input_len]
        if pad_token_id >= 0:
            valid_positions = row.ne(pad_token_id).nonzero(as_tuple=False)
            valid_len = int(valid_positions[-1].item() + 1) if valid_positions.numel() else 0
        else:
            valid_len = row.numel()
        valid_len = min(valid_len, max_input_len)
        if valid_len < 2:
            continue
        seq = row[:valid_len].detach().tolist()
        seq_lengths.append(valid_len)
        seq_id = next_seq_id
        next_seq_id += 1
        token_values = seq[:-1]
        label_values = seq[1:]
        offset = 0
        while offset < len(token_values):
            if len(cur_tokens) == micro_seq_length:
                flush_chunk()
            take = min(micro_seq_length - len(cur_tokens),
                       len(token_values) - offset)
            for i in range(take):
                pos = offset + i
                cur_tokens.append(token_values[pos])
                cur_labels.append(label_values[pos])
                cur_loss_mask.append(1.0)
                cur_position_ids.append(pos)
                cur_seq_ids.append(seq_id)
                cur_positions.append(pos)
            offset += take

    flush_chunk()

    if not chunks:
        chunks.append(_empty_packed_slice(micro_seq_length, pad_value, device))
        loss_masks.append(torch.zeros((1, micro_seq_length),
                                      dtype=torch.float, device=device))

    if pad_chunks_to_multiple > 1:
        while len(chunks) < pad_chunks_to_multiple or len(chunks) % pad_chunks_to_multiple:
            chunks.append(_empty_packed_slice(micro_seq_length, pad_value, device,
                                             past_length=len(history_seq_ids)))
            loss_masks.append(torch.zeros((1, micro_seq_length),
                                          dtype=torch.float, device=device))
            history_seq_ids.extend([-1] * micro_seq_length)
            history_positions.extend([-1] * micro_seq_length)

    # DEBUG for variable length training.
    debug_limit = getattr(args, 'variable_seq_debug_num_batches', 0)
    debug_count = getattr(get_packed_variable_sliced_batch, '_debug_count', 0)
    if debug_limit > debug_count:
        print_rank_0(
            "[variable-seq][pack] "
            f"microbatch={debug_count}, seq_lengths={seq_lengths}, "
            f"chunk_size={micro_seq_length}, chunks={len(chunks)}, "
            f"chunk_segments={chunk_segments}"
        )
        get_packed_variable_sliced_batch._debug_count = debug_count + 1

    return chunks, torch.cat(loss_masks, dim=0)


def _summarize_packed_segments(seq_ids, positions):
    segments = []
    cur_seq_id = None
    start = None
    length = 0
    for seq_id, pos in zip(seq_ids, positions):
        if seq_id < 0:
            continue
        if seq_id != cur_seq_id or (start is not None and pos != start + length):
            if cur_seq_id is not None:
                segments.append((cur_seq_id, start, length))
            cur_seq_id = seq_id
            start = pos
            length = 1
        else:
            length += 1
    if cur_seq_id is not None:
        segments.append((cur_seq_id, start, length))
    return segments


def _empty_packed_slice(micro_seq_length, pad_value, device, past_length=0):
    tokens = torch.full((1, micro_seq_length), pad_value,
                        dtype=torch.long, device=device)
    labels = torch.full((1, micro_seq_length), pad_value,
                        dtype=torch.long, device=device)
    position_ids = torch.zeros((1, micro_seq_length),
                               dtype=torch.long, device=device)
    attention_mask = torch.ones((1, 1, micro_seq_length,
                                 past_length + micro_seq_length),
                                dtype=torch.bool, device=device)
    return tokens, position_ids, attention_mask, labels


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)
