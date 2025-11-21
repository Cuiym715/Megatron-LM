import einops
import itertools
import torch

from . import llama_reference
from megatron import print_rank_last
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, _gather_along_last_dim
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.transformer import NoopTransformerLayer
from megatron.utils import unwrap_model
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


llama = llama_reference.Llama()


def get_tensor_broadcast_pp(state_dict, key, shape, dtype):
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    key_matches = key in state_dict
    if key_matches:
        assert state_dict[key].shape == shape and state_dict[key].dtype == dtype, f"{state_dict[key].shape} vs {shape}, {state_dict[key].dtype} vs {dtype}"
        tensor = state_dict[key].contiguous()
    else:
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
    num_matches = torch.tensor(int(key_matches), device="cuda")
    torch.distributed.all_reduce(num_matches, group=pp_group)
    num_matches = num_matches.item()
    assert num_matches == 1, f"{key=} {num_matches=}"
    src_rank = torch.tensor(int(pp_rank if key_matches else 0), device="cuda")
    torch.distributed.all_reduce(src_rank, group=pp_group)
    src_rank = src_rank.item()
    torch.distributed.broadcast(tensor, src=torch.distributed.get_global_rank(pp_group, src_rank), group=pp_group)
    return tensor


def get_state_dict(args, unwrapped_model, layer_number):
    h = args.hidden_size
    a = args.num_attention_heads
    g = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    d = h // a
    tp_size = args.tensor_model_parallel_size
    ep_size = args.expert_model_parallel_size
    cp_overlap = args.context_parallel_comm_overlap_gemm and args.context_parallel_size >= 2
    layers = itertools.chain.from_iterable([m.language_model.encoder.layers for m in unwrapped_model])
    layers = [layer for layer in layers if not isinstance(layer, NoopTransformerLayer) and layer.layer_number == layer_number]
    assert len(layers) <= 1
    s = layers[0].state_dict() if layers else dict()
    is_moe_layer = args.num_experts is not None and (layer_number - args.moe_first) % args.moe_layer_interval == 0
    shape_dict = {
        "input_layernorm.weight": (h,),
        "self_attention.query_key_value.weight": ((a + g + g) // tp_size * d, h),
        "self_attention.dense.weight": (h, h // tp_size),
        "post_attention_layernorm.weight": (h,),
    }
    if is_moe_layer:
        H = args.expert_hidden_size
        num_local_experts = args.num_experts // ep_size
        shape_dict.update({
            "mlp.router.weight": (args.num_experts, h),
            **dict(
                [
                    (f"mlp.experts.weight1", (h, num_local_experts * (2 * H // tp_size))),
                    (f"mlp.experts.weight2", (num_local_experts * (H // tp_size), h)),
                ]
                if args.moe_grouped_gemm else
                itertools.chain.from_iterable([
                    (f"mlp.experts.linear_fc1.weight{i}", (2 * H // tp_size, h)),
                    (f"mlp.experts.linear_fc2.weight{i}", (h, H // tp_size)),
                ] for i in range(num_local_experts))
                if args.moe_te_grouped_gemm else
                itertools.chain.from_iterable([
                    (f"mlp.experts.local_experts.{i}.dense_h_to_4h.weight", (2 * H // tp_size, h)),
                    (f"mlp.experts.local_experts.{i}.dense_4h_to_h.weight", (h, H // tp_size)),
                ] for i in range(num_local_experts))
            ),
        })
        if args.shared_expert_hidden_size:
            shape_dict.update({
                "mlp.shared_expert.dense_h_to_4h.weight": (2 * args.shared_expert_hidden_size // tp_size, h),
                "mlp.shared_expert.dense_4h_to_h.weight": (h, args.shared_expert_hidden_size // tp_size),
            })
            if args.shared_expert_combine_method == "softmax":
                shape_dict.update({
                    "mlp.coefficient.weight": (2, h),
                    "mlp.coefficient.bias": (2,),
                })
    else:
        H = args.ffn_hidden_size
        shape_dict.update({
            "mlp.dense_h_to_4h.weight": (2 * H // tp_size, h),
            "mlp.dense_4h_to_h.weight": (h, H // tp_size),
        })
    s = {k: get_tensor_broadcast_pp(s, k, shape, args.params_dtype) for k, shape, in shape_dict.items()}
    if is_moe_layer:
        if args.moe_grouped_gemm:
            w1 = s.pop("mlp.experts.weight1")
            w2 = s.pop("mlp.experts.weight2")
            for i in range(num_local_experts):
                s[f"mlp.experts.local_experts.{i}.dense_h_to_4h.weight"] = w1.view(num_local_experts, h, -1)[i].T
                s[f"mlp.experts.local_experts.{i}.dense_4h_to_h.weight"] = w2.view(num_local_experts, -1, h)[i].T
        elif args.moe_te_grouped_gemm:
            for i in range(num_local_experts):
                s[f"mlp.experts.local_experts.{i}.dense_h_to_4h.weight"] = s.pop(f"mlp.experts.linear_fc1.weight{i}")
                s[f"mlp.experts.local_experts.{i}.dense_4h_to_h.weight"] = s.pop(f"mlp.experts.linear_fc2.weight{i}")
        for i in range(num_local_experts):
            for key in ["dense_h_to_4h.weight", "dense_4h_to_h.weight"]:
                input_tensor = s[f"mlp.experts.local_experts.{i}.{key}"].contiguous()
                output_tensor = torch.empty((ep_size, *input_tensor.shape), dtype=input_tensor.dtype, device=input_tensor.device)
                torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=mpu.get_expert_model_parallel_group())
                for ep_rank in range(ep_size):
                    s[f"mlp.experts.local_experts.{i + ep_rank * num_local_experts}.{key}"] = output_tensor[ep_rank]
    res = dict([
        (f"input_layernorm.weight", s["input_layernorm.weight"]),
        (f"self_attn.q_proj.weight", _gather_along_first_dim(
            einops.rearrange(s["self_attention.query_key_value.weight"], "(agg d) h -> agg d h", d=d)[:a // tp_size].flatten(0, -2) if cp_overlap else
            einops.rearrange(s["self_attention.query_key_value.weight"], "(g ga d) h -> g ga d h", ga=a // g + 2, d=d)[:, :-2].flatten(0, -2))),
        (f"self_attn.k_proj.weight", _gather_along_first_dim(
            einops.rearrange(s["self_attention.query_key_value.weight"], "(agg d) h -> agg d h", d=d)[a // tp_size:(a + g) // tp_size].flatten(0, -2) if cp_overlap else
            einops.rearrange(s["self_attention.query_key_value.weight"], "(g ga d) h -> g ga d h", ga=a // g + 2, d=d)[:, -2].flatten(0, -2))),
        (f"self_attn.v_proj.weight", _gather_along_first_dim(
            einops.rearrange(s["self_attention.query_key_value.weight"], "(agg d) h -> agg d h", d=d)[(a + g) // tp_size:].flatten(0, -2) if cp_overlap else
            einops.rearrange(s["self_attention.query_key_value.weight"], "(g ga d) h -> g ga d h", ga=a // g + 2, d=d)[:, -1].flatten(0, -2))),
        (f"self_attn.o_proj.weight", _gather_along_last_dim(s["self_attention.dense.weight"])),
        (f"post_attention_layernorm.weight", s["post_attention_layernorm.weight"]),
    ])
    if is_moe_layer:
        res.update(dict([
            ("mlp.router.weight", s["mlp.router.weight"]),
            *itertools.chain.from_iterable([
                (f"mlp.experts.{i}.gate_proj.weight", _gather_along_first_dim(einops.rearrange(
                    s[f"mlp.experts.local_experts.{i}.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[0].contiguous())),
                (f"mlp.experts.{i}.up_proj.weight", _gather_along_first_dim(einops.rearrange(
                    s[f"mlp.experts.local_experts.{i}.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[1].contiguous())),
                (f"mlp.experts.{i}.down_proj.weight", _gather_along_last_dim(s[f"mlp.experts.local_experts.{i}.dense_4h_to_h.weight"])),
            ] for i in range(args.num_experts)),
        ]))
        if args.shared_expert_hidden_size:
            res.update(dict([
                (f"mlp.shared_expert.gate_proj.weight", _gather_along_first_dim(einops.rearrange(
                    s[f"mlp.shared_expert.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[0].contiguous())),
                (f"mlp.shared_expert.up_proj.weight", _gather_along_first_dim(einops.rearrange(
                    s[f"mlp.shared_expert.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[1].contiguous())),
                (f"mlp.shared_expert.down_proj.weight", _gather_along_last_dim(s[f"mlp.shared_expert.dense_4h_to_h.weight"])),
            ]))
            if args.shared_expert_combine_method == "softmax":
                res.update(dict([
                    ("mlp.coefficient.weight", s["mlp.coefficient.weight"]),
                    ("mlp.coefficient.bias", s["mlp.coefficient.bias"]),
                ]))
    else:
        res.update(dict([
            (f"mlp.gate_proj.weight", _gather_along_first_dim(einops.rearrange(s["mlp.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[0])),
            (f"mlp.up_proj.weight", _gather_along_first_dim(einops.rearrange(s["mlp.dense_h_to_4h.weight"], "(i2 H) h -> i2 H h", i2=2)[1])),
            (f"mlp.down_proj.weight", _gather_along_last_dim(s["mlp.dense_4h_to_h.weight"])),
        ]))
    return res


def setup(args, model, writer):
    torch.distributed.barrier()
    print_rank_last("Golden model: broadcast weights")
    g = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    tp_size = args.tensor_model_parallel_size
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    s = unwrapped_model[0].state_dict()
    s_last = unwrapped_model[-1].state_dict()
    state_dict_reference = {
        "model.embed_tokens.weight": _gather_along_first_dim(get_tensor_broadcast_pp(s, "language_model.embedding.word_embeddings.weight", (args.padded_vocab_size // tp_size, args.hidden_size), args.params_dtype))[:args.trained_vocab_size],
        **dict(
            itertools.chain.from_iterable([
                (f"model.layers.{i}." + k, v)
                for k, v in get_state_dict(args, unwrapped_model, i + 1).items()
            ] for i in range(args.num_layers))
        ),
        "model.norm.weight": get_tensor_broadcast_pp(s_last, "language_model.encoder.final_layernorm.weight", (args.hidden_size,), args.params_dtype),
        "lm_head.weight": _gather_along_first_dim(get_tensor_broadcast_pp(s_last, "language_model.output_layer.weight", (args.padded_vocab_size // tp_size, args.hidden_size), args.params_dtype))[:args.trained_vocab_size],
    }
    if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
        print("Golden model: init model")
        llama.init_model(
            vocab_size=args.trained_vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.ffn_hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=g,
            max_position_embeddings=args.seq_length,
            rms_norm_eps=args.layernorm_epsilon,
            rope_theta=args.rope_theta,
            num_experts=args.num_experts,
            expert_hidden_size=args.expert_hidden_size,
            shared_expert_hidden_size=args.shared_expert_hidden_size,
            shared_expert_combine_method=args.shared_expert_combine_method,
            moe_layer_interval=args.moe_layer_interval,
            moe_first=args.moe_first,
            moe_router_topk=args.moe_router_topk,
            moe_aux_loss_coeff=args.moe_aux_loss_coeff,
            moe_expert_capacity_factor=args.moe_expert_capacity_factor,
            moe_token_drop_policy=args.moe_token_drop_policy,
            torch_dtype=args.params_dtype,
        )
        print("Golden model: load state dict")
        llama.model.load_state_dict(state_dict_reference)
        print("Golden model: init optimizer")
        optimizer_kwargs = {
            "sgd": dict(momentum=args.sgd_momentum),
            "adam": dict(betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps),
        }[args.optimizer]
        num_microbatches = args.global_batch_size / args.micro_batch_size
        llama.init_optimizer(args.lr, args.min_lr, args.lr_decay_iters, args.lr_warmup_iters, args.weight_decay, args.optimizer, optimizer_kwargs, num_microbatches, args.clip_grad or None)
        llama.init_tensorboard(writer, args.tensorboard_log_interval)
        print("Golden model: finish init")


def feed_input_data(input_ids, labels):
    if not mpu.is_pipeline_last_stage():
        return
    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    dp_group = mpu.get_data_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()
    for src_rank in range(0, dp_size, cp_size):
        if src_rank == dp_rank:
            cur_input_ids = input_ids.contiguous()
            cur_labels = labels.contiguous()
        else:
            cur_input_ids = torch.empty_like(input_ids)
            cur_labels = torch.empty_like(labels)
        torch.distributed.broadcast(cur_input_ids, src=torch.distributed.get_global_rank(dp_group, src_rank), group=dp_group)
        torch.distributed.broadcast(cur_labels, src=torch.distributed.get_global_rank(dp_group, src_rank), group=dp_group)
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            llama.feed_input_data(cur_input_ids, cur_labels)


def step():
    if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
        llama.step()
