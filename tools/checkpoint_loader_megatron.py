from functools import partial
import json
import os
import sys
import types

import torch
import numpy as np


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def update_model_weights(model, optim_model, dtype):
    """
    Use parameters in optim_model to update parameters in model and keep parameters in model.dtype
    """
    for ((name1, param1), (name2, param2)) in zip(model.named_parameters(), optim_model.named_parameters()):
        if param1.requires_grad:
            converted_param2 = param2.data.to(dtype)
            param1.data.copy_(converted_param2)


def _check_model_equal(model1: torch.nn.Module, model2: torch.nn.Module, ep_rank, dtype=torch.float16, atol=5e-3, rtol=5e-3):
    for i, ((name1, param1), (name2, param2)) in enumerate(zip(model1.named_parameters(), model2.named_parameters())):
        assert param1.requires_grad == param2.requires_grad, f"Parameter_{i}_{name1}.requires_grad mismatch"
        
        is_dense_param = getattr(param1, 'allreduce', True)
        if param1.requires_grad and (ep_rank == 0 or not is_dense_param):
            assert param1.data.nelement() == param2.data.nelement(), f"Parameter_{i}_{name1}.numel() mismatch"
            converted_param1 = param1.data.to(dtype)
            converted_param2 = param2.data.to(dtype)
            if not torch.allclose(converted_param1, converted_param2, atol=atol, rtol=rtol):
                diff = torch.abs(converted_param1 - converted_param2)
                max_diff = torch.max(diff)
                mean_diff = torch.mean(diff)
                print(f"Parameter_{i}_{name1} max_diff {max_diff}, mean_diff {mean_diff}")
                print(f"Parameter_{i}_{name1} shape: {param1.data.shape}, dtype: {dtype}")


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint, _load_base_checkpoint
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType

        from checkpoint_util_megatron import DummyOptimizer
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--make-main-grad-addresss-divisible-by', '1',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--sequence-parallel',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs, checkpoint_args = load_args_from_checkpoint(margs)
    intermediate_storage_path = args.universal_checkpoint_dir
    os.makedirs(intermediate_storage_path, exist_ok=True)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * \
        margs.pipeline_model_parallel_size

    # keep dtype consistent
    if not hasattr(margs, 'params_dtype'):
        margs.params_dtype = torch.bfloat16
        checkpoint_args.accumulate_allreduce_grads_in_fp32 = True
        margs.accumulate_allreduce_grads_in_fp32 = True
    margs.bf16 = margs.params_dtype == torch.bfloat16
    margs.fp16 = margs.params_dtype == torch.half

    # TODO: support accumulate_allreduce_grads_in_fp32 = False
    assert (checkpoint_args.accumulate_allreduce_grads_in_fp32), \
        "accumulate_allreduce_grads_in_fp32 must be True"

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(
                    f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('add_position_embedding', True)
    check_for_arg('use_rotary_position_embeddings', False)
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # 优化此处 loader model tokenizer.vocab_size 的处理, 方式改为直接传参
    if args.vocab_size is not None:
        print(f'loader vocab_size = {args.vocab_size}')
        margs.vocab_size = args.vocab_size
    else:
        margs.vocab_size = None

    # Group Query Attention
    previous_num_query_groups = margs.num_query_groups if margs.group_query_attention else margs.num_attention_heads

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'LLAMA':
        from pretrain_llama import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    load_optimizer = args.process_optimizer
    if load_optimizer:
        # set args to load optimizer state
        # see: megatron/checkpointing.py:load_checkpoint()
        margs.use_distributed_optimizer = True
        margs.no_load_optim = False

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = getattr(
        checkpoint_args, 'consumed_train_samples', 0)
    consumed_valid_samples = getattr(
        checkpoint_args, 'consumed_valid_samples', 0)

    def get_models_and_load(pp_rank, vp_size, ep_rank, tp_size, dtype, load_optimizer):
        """
            param:
                vp_size -- virtual pipeline parallel size
                tp_size -- tp size
                dtype -- params dtype
            return:
                models -- models[vp_rank][tp_rank] is a model
                optim_models -- optim_models[key][vp_rank][tp_rank] is a model
        """
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        mpu.set_expert_model_parallel_rank(ep_rank)
        # get models
        from checkpoint_util_megatron import get_models
        # OOM: 同时创建一个 tp_group 的模型, 理论上 80 * 8 = 640GB < 1000GB 永远不会 OOM
        models = get_models(model_provider, pp_rank, vp_size, tp_size, dtype)
        optim_models = {}
        if load_optimizer:
            for key in ("param", "exp_avg", "exp_avg_sq"):
                optim_models[key] = get_models(
                    model_provider, pp_rank, vp_size, tp_size, torch.float32)
        # load model and optimizer from checkpoint
        for tp_rank in range(tp_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            model_ = []
            for vp_rank in range(vp_size):
                model_.append(models[vp_rank][tp_rank])

            optimizer = None
            optim_models_ = {}
            if load_optimizer:
                for key, value in optim_models.items():
                    optim_models_[key] = []
                    for vp_rank in range(vp_size):
                        optim_models_[key].append(value[vp_rank][tp_rank])
                optimizer = DummyOptimizer(optim_models_)

            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint(model_, optimizer=optimizer,
                            opt_param_scheduler=None)
            # check value consistency
            nonlocal consumed_train_samples
            nonlocal consumed_valid_samples
            print(f"margs.consumed_train_samples = {margs.consumed_train_samples}, consumed_train_samples = {consumed_train_samples}")
            # check diff between model and optimizer model
            if load_optimizer:
                for src_model, optim_model in zip(model_, optim_models_["param"]):
                    update_model_weights(src_model, optim_model, dtype)
                    _check_model_equal(src_model, optim_model, ep_rank, dtype)

        return models, optim_models

    set_global_variables(margs)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    # set fake cp for the construction of RotaryEmbedding
    mpu.set_context_parallel_world_size(1)
    mpu.set_context_parallel_rank(0)

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1
    total_meta_info = {}
    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.tokenizer_model = margs.tokenizer_model
    md.rms_norm = margs.rms_norm
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embeddings = margs.add_position_embedding
    md.linear_bias = margs.add_bias_linear
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = tp_size
    md.previous_expert_parallel_size = ep_size
    md.previous_pipeline_parallel_size = pp_size
    md.previous_virtual_pipeline_parallel_size = vp_size
    md.vocab_size = args.vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.kaimm_num_layers_padding_front = margs.kaimm_num_layers_padding_front
    md.kaimm_num_layers_padding_back = margs.kaimm_num_layers_padding_back
    md.previous_num_query_groups = previous_num_query_groups
    md.checkpoint_args = checkpoint_args

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def namespace_to_dict(namespace):
        def convert(obj):
            # 如果对象是 NumPy 数组, 转换为列表
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # 如果对象是 NumPy 数据类型, 转换为相应的 Python 类型
            elif isinstance(obj, np.generic):
                return obj.item()
            # 对于其它类型, 直接返回
            return obj
        
        # 使用字典推导式, 并对每个值应用 convert 函数
        return {k: convert(v) for k, v in vars(namespace).items()}
    # md_dict = namespace_to_dict(md)
    # total_meta_info['metadata'] = md_dict

    def save_msg_to_ceph(msg, msg_name,intermediate_storage_path):
        msg_path = f"{intermediate_storage_path}/{msg_name}.pt"
        torch.save(msg, msg_path)
        return msg_path

    def save_total_meta_info(total_meta_info,intermediate_storage_path):
        meta_info_path = f"{intermediate_storage_path}/total_meta_info.json"
        with open(meta_info_path, 'w') as f:
            json.dump(total_meta_info, f)

    def queue_put(category, name, msg,intermediate_storage_path,total_meta_info):
        real_name = name if category is None or category == "model" else f"{category} {name}"
        print(f"sending {real_name}")
        msg_path = save_msg_to_ceph(msg, real_name,intermediate_storage_path)
        meta_info = {"name": real_name, "path": msg_path}
        total_meta_info[real_name] = msg_path
        queue.put(meta_info)

    import gc
    # load all models
    # OOM: 一次性创建整个模型参数导致 OOM
    # model[vp_rank][tp_rank] is a model
    for pp_rank in range(pp_size):
        for ep_rank in range(ep_size):
            print(f"start loading models at pp_rank {pp_rank}")
            model, optim = get_models_and_load(pp_rank, vp_size, ep_rank, tp_size, md.params_dtype, load_optimizer=load_optimizer)
            print("finish get_models_and_load")
            merge_and_send_models(model, pp_rank, ep_rank, md, tp_size, pp_size, vp_size,
                                partial(queue_put, "model", intermediate_storage_path=intermediate_storage_path, total_meta_info=total_meta_info))
            print("finish merge_and_send_models")
            del model
            gc.collect()
            print("del model")
            for key, value in optim.items():
                print(f"process: {key}") 
                merge_and_send_models(value, pp_rank, ep_rank, md, tp_size, pp_size, vp_size,
                                partial(queue_put, f"optim-{key}", intermediate_storage_path=intermediate_storage_path, total_meta_info=total_meta_info))
                print(f"optim {key} merge_and_send_models") 
            del optim
            gc.collect()
            print("del optim")
            print(f"pp_rank {pp_rank} finished")
    # Load and send common checkpoint data
    if load_optimizer:
        base_ckp, _ = _load_base_checkpoint(args.load_dir, rank0=True)
        msg = {}
        for key in {'optimizer', 'opt_param_scheduler'}:
            msg[key] = base_ckp.get(key, None)
        queue_put(category=None, name="base_checkpoint", msg=msg, intermediate_storage_path=intermediate_storage_path, total_meta_info=total_meta_info)

    save_total_meta_info(total_meta_info,intermediate_storage_path)
    queue.put("done")


def merge_and_send_models(pp_model, pp_rank, ep_rank, md, tp_size, pp_size, vp_size, queue_put):
    """
        merge models in tp dimension and send to queue
    """
    from megatron.core.transformer.moe.moe_layer import BaseMoELayer
    from megatron.model.transformer import NoopTransformerLayer

    # Send embeddings
    # OOM: 目前待发送的模型参数在合并后发送在消息队列中, 若saver消费慢, 或需要重复访问会导致消息堆积占用大量内存导致 OOM
    if(pp_rank == 0):
        models = pp_model[0]
        message = {
            "word embeddings": torch.cat(
                [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(
                    tp_size)],
                dim=0)
        }
        if md.position_embeddings:
            message["position embeddings"] = models[0].language_model.embedding.position_embeddings.weight.data

        if ep_rank == 0:
            queue_put("embeddings", message)

    for vp_rank in range(vp_size):
        models = pp_model[vp_rank]
        num_layers = len(models[0].language_model.encoder.layers)
        for layer_num in range(num_layers):
            message = {}
            # Get non-parallel tensors from tp_rank 0
            layer = models[0].language_model.encoder.layers[layer_num]
            if isinstance(layer, NoopTransformerLayer):
                continue
            total_layer_num = layer.layer_number - 1
            is_moe_layer = isinstance(layer.mlp, BaseMoELayer)
            message["input layernorm weight"] = layer.input_layernorm.weight.data
            message["input layernorm bias"] = layer.input_layernorm.bias.data \
                if hasattr(layer.input_layernorm, 'bias') else None
            message["post layernorm weight"] = layer.post_attention_layernorm.weight.data
            message["post layernorm bias"] = layer.post_attention_layernorm.bias.data \
                if hasattr(layer.post_attention_layernorm, 'bias') else None
            if md.linear_bias:
                message["dense bias"] = layer.self_attention.dense.bias.data
                message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data
            if is_moe_layer:
                message["router weight"] = layer.mlp.router.weight.data
                if md.checkpoint_args.shared_expert_hidden_size and md.checkpoint_args.shared_expert_combine_method == "softmax":
                    message["moe coefficient weight"] = layer.mlp.coefficient.weight.data
                    message["moe coefficient bias"] = layer.mlp.coefficient.bias.data

            # Grab all parallel tensors for this layer
            qkv_weight = []
            qkv_bias = []
            dense_weight = []
            mlp_l0_weight = []
            mlp_l0_bias = []
            mlp_l1_weight = []
            if is_moe_layer:
                experts_weight = {local_expert_index: {
                    "mlp_l0_weight": [],
                    "mlp_l0_bias": [],
                    "mlp_l1_weight": [],
                } for local_expert_index in layer.mlp.local_expert_indices}

            def mlp_to_mlp_weight(mlp, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight):
                mlp_l0_weight.append(mlp.dense_h_to_4h.weight.data)
                mlp_l1_weight.append(mlp.dense_4h_to_h.weight.data)
                if md.linear_bias:
                    mlp_l0_bias.append(mlp.dense_h_to_4h.bias.data)

            for tp_rank, model in enumerate(models):
                layer = model.language_model.encoder.layers[layer_num]
                qkv_weight.append(
                    layer.self_attention.query_key_value.weight.data)
                dense_weight.append(layer.self_attention.dense.weight.data)
                if md.linear_bias:
                    qkv_bias.append(
                        layer.self_attention.query_key_value.bias.data)
                if is_moe_layer:
                    if md.checkpoint_args.moe_grouped_gemm:
                        if md.linear_bias:
                            raise NotImplementedError("bias in the expert layer is not supported in Grouped GEMM yet")
                        w1 = layer.mlp.experts.weight1.view(layer.mlp.experts.num_local_experts, layer.mlp.experts.config.hidden_size, -1)
                        w2 = layer.mlp.experts.weight2.view(layer.mlp.experts.num_local_experts, -1, layer.mlp.experts.config.hidden_size)
                        for local_expert_index, mlp_l0_weight_T, mlp_l1_weight_T in zip(layer.mlp.local_expert_indices, w1, w2):
                            experts_weight[local_expert_index]["mlp_l0_weight"].append(mlp_l0_weight_T.T.data)
                            experts_weight[local_expert_index]["mlp_l1_weight"].append(mlp_l1_weight_T.T.data)
                    elif md.checkpoint_args.moe_te_grouped_gemm:
                        if md.linear_bias:
                            raise NotImplementedError("bias in the expert layer is not supported in Grouped GEMM yet")
                        for i, local_expert_index in enumerate(layer.mlp.local_expert_indices):
                            experts_weight[local_expert_index]["mlp_l0_weight"].append(getattr(layer.mlp.experts.linear_fc1, f"weight{i}").data)
                            experts_weight[local_expert_index]["mlp_l1_weight"].append(getattr(layer.mlp.experts.linear_fc2, f"weight{i}").data)
                    else:
                        for expert, local_expert_index in zip(layer.mlp.experts.local_experts, layer.mlp.local_expert_indices):
                            mlp_to_mlp_weight(expert,
                                experts_weight[local_expert_index]["mlp_l0_weight"],
                                experts_weight[local_expert_index]["mlp_l0_bias"],
                                experts_weight[local_expert_index]["mlp_l1_weight"])
                    if md.checkpoint_args.shared_expert_hidden_size:
                        mlp_to_mlp_weight(layer.mlp.shared_expert, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight)
                else:
                    mlp_to_mlp_weight(layer.mlp, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight)

            def mlp_weight_to_message(mlp_l0_weight, mlp_l0_bias, mlp_l1_weight):
                message = dict()
                # Handle gated linear units
                if md.swiglu:
                    # concat all the first halves ('W's) and all the second halves ('V's)
                    for tp_rank in range(tp_size):
                        mlp_l0_weight[tp_rank] = torch.chunk(
                            mlp_l0_weight[tp_rank], 2, dim=0)
                    message["mlp l0 weight W"] = torch.cat(
                        [w[0] for w in mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.cat(
                        [w[1] for w in mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)
                message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
                if md.linear_bias:
                    if md.swiglu:
                        for tp_rank in range(tp_size):
                            mlp_l0_bias[tp_rank] = torch.chunk(
                                mlp_l0_bias[tp_rank], 2, dim=0)
                        message["mlp l0 bias W"] = torch.cat(
                            [b[0] for b in mlp_l0_bias], dim=0)
                        message["mlp l0 bias V"] = torch.cat(
                            [b[1] for b in mlp_l0_bias], dim=0)
                    else:
                        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
                return message

            if is_moe_layer:
                for local_expert_index in layer.mlp.local_expert_indices:
                    expert_message = mlp_weight_to_message(
                        experts_weight[local_expert_index]["mlp_l0_weight"],
                        experts_weight[local_expert_index]["mlp_l0_bias"],
                        experts_weight[local_expert_index]["mlp_l1_weight"])
                    queue_put(f"transformer layer {total_layer_num} expert {local_expert_index}", expert_message)
                if md.checkpoint_args.shared_expert_hidden_size:
                    expert_message = mlp_weight_to_message(mlp_l0_weight, mlp_l0_bias, mlp_l1_weight)
                    message.update({
                        f"moe shared_expert {k}": v
                        for k, v in expert_message.items()
                    })
            else:
                message.update(mlp_weight_to_message(mlp_l0_weight, mlp_l0_bias, mlp_l1_weight))

            # simple concat of the rest
            message["qkv weight"] = torch.cat(qkv_weight, dim=0)
            message["dense weight"] = torch.cat(dense_weight, dim=1)
            if md.linear_bias:
                message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            if ep_rank == 0:
                queue_put(f"transformer layer {total_layer_num}", message)

    # Send final layernorm from tp_rank 0
    if(pp_rank == pp_size -1):
        message = {
            "weight": models[0].language_model.encoder.final_layernorm.weight.data,
            "bias": models[0].language_model.encoder.final_layernorm.bias.data
            if hasattr(models[0].language_model.encoder.final_layernorm, 'bias') else None
        }
        if ep_rank == 0:
            queue_put("final layernorm", message)

        if md.output_layer:
            message = {
                "weight": torch.cat(
                    [models[tp_rank].language_model.output_layer.weight.data for tp_rank in range(
                        tp_size)],
                    dim=0)
            }
            if ep_rank == 0:
                queue_put("output layer", message)

        # Send BERT lm head and binary head if it exists
        if md.model_type == 'BERT':
            message = {
                "weight": models[0].language_model.pooler.dense.weight.data,
                "bias": models[0].language_model.pooler.dense.bias.data
            }
            if ep_rank == 0:
                queue_put("pooler", message)

            message = {
                "dense weight": models[0].lm_head.dense.weight.data,
                "dense bias": models[0].lm_head.dense.bias.data,
                "layernorm weight": models[0].lm_head.layernorm.weight.data,
                "layernorm bias": models[0].lm_head.layernorm.bias.data
            }
            if ep_rank == 0:
                queue_put("lm head", message)

            if md.bert_binary_head:
                message = {
                    "weight": models[0].binary_head.weight.data,
                    "bias": models[0].binary_head.bias.data
                }
                if ep_rank == 0:
                    queue_put("binary head", message)
    


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
