# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import reference
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import LlamaModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_sliced_batch
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()

    print_rank_0('building GPT model ...')
    model = LlamaModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return partial(get_sliced_batch, tokens, position_ids, attention_mask, labels), partial(loss_func, loss_mask)


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(batch, kv_cache, model):
    """Forward step."""
    args = get_args()
    tokens, position_ids, attention_mask, labels = batch
    if args.train_golden_model:
        reference.adapter.feed_input_data(tokens, labels)

    output_tensor = model(tokens, position_ids, attention_mask, kv_cache=kv_cache,
                          labels=labels)

    return output_tensor


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             get_batch,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
