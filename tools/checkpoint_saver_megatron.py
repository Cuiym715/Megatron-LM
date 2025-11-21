from collections import defaultdict
from functools import partial
import os
import sys

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target pipeline model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-virtual-pipeline-parallel-size', type=int,
                       help='Target virtual pipeline model parallel size, default to the virtual pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-expert-parallel-size', type=int,
                       help='Target expert model parallel size, default to the expert model parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-num-query-groups', type=int,
                       help='Target number of key_value heads that should be used to implement Grouped Query Attention. When converting '
                       'a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling '
                       'all the original heads within that group. For more details checkout https://arxiv.org/pdf/2305.13245.pdf. '
                       'If it is not specified, will default to the num_query_groups in the input checkpoint if provided by the loader, '
                       'otherwise to num_attention_heads.')
    group.add_argument('--target-moe-grouped-gemm', type=int,
                       help='Use MoE grouped gemm in the target model. Zero for false, other values for true')
    group.add_argument('--target-moe-te-grouped-gemm', type=int,
                       help='Use MoE grouped gemm from transformer_engine in the target model. Zero for false, other values for true')
    group.add_argument('--target-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--target-kaimm-num-layers-padding-front', type=int, default=None,
                       help='Number of padding layers between the embedding layer '
                       'and the first transformer layer')
    group.add_argument('--target-kaimm-num-layers-padding-back', type=int, default=None,
                       help='Number of padding layers between the LM head layer '
                       'and the last transformer layer')

class BufferedQueue:
    """
    A queue that allows you to get messages out of order, but will buffer them until they are ready.
    It is not thread safe.
    """

    def __init__(self, queue):
        self._queue = queue
        self._buffer = {}
        self._repeated_times = defaultdict(int)
        self._done = False

    def get_obj(self):
        assert len(self._buffer) == 0
        assert not self._done
        msg = self._queue.get()
        if msg == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        return msg

    def get(self, name, repeat):
        # repeat: Time to repeat the msg in buffer. Dense weights are buffered for EP times.
        if name in self._buffer:
            msg = self._buffer[name]
            self._repeated_times[name] += 1
            if repeat == self._repeated_times[name]:
                self._buffer.pop(name)
            return msg
        while not self._done:
            msg = self._queue.get()
            if msg == "exit":
                print("Loader exited, exiting saver")
                exit(1)
            if msg == "done":
                self._done = True
                break
            assert isinstance(msg, dict)
            if msg["name"] == name:
                if repeat > 1:
                    self._repeated_times[name] = 1
                    self._buffer[name] = msg
                return msg
            else:
                self._buffer[msg["name"]] = msg
        if name == "done" and self._done:
            return "done"
        return None


def _megatron_convert_group_query_attentions(qkv_weights, previous_num_query_groups, target_num_query_groups, num_heads, kv_channels):
    """
    When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed 
    by meanpooling all the original heads within that group. For more details checkout https://arxiv.org/pdf/2305.13245.pdf.
    """
    if previous_num_query_groups == target_num_query_groups:
        return qkv_weights
    # stored in [ng * (nh//ng+2) * kv_channels, :]
    input_shape = qkv_weights.shape
    hidden_size = input_shape[-1]
    saved_shape = (previous_num_query_groups, (num_heads//previous_num_query_groups + 2) * kv_channels) + input_shape[1:]
    q,k,v = qkv_weights.view(*saved_shape).split([(num_heads//previous_num_query_groups)*kv_channels, kv_channels, kv_channels], dim=1)
    kv_shape = k.shape
    # print(f"input_shape:{input_shape}, saved_shape:{saved_shape}, kv_shape:{kv_shape}")
    if previous_num_query_groups > target_num_query_groups:
        assert previous_num_query_groups % target_num_query_groups == 0
        new_kv_shape = (target_num_query_groups, previous_num_query_groups//target_num_query_groups) + kv_shape[1:]
        k = k.reshape(new_kv_shape).mean(dim=1)
        v = v.reshape(new_kv_shape).mean(dim=1)
    else:
        assert target_num_query_groups % previous_num_query_groups == 0
        k = k.repeat_interleave(target_num_query_groups//previous_num_query_groups, dim=0)
        v = v.repeat_interleave(target_num_query_groups//previous_num_query_groups, dim=0)

    qkv = torch.cat([
        q.reshape((target_num_query_groups, -1, hidden_size)),
        k.reshape((target_num_query_groups, -1, hidden_size)),
        v.reshape((target_num_query_groups, -1, hidden_size)),
    ], dim=1).reshape((-1, hidden_size)).contiguous()

    assert qkv.shape == torch.Size((target_num_query_groups*(num_heads//target_num_query_groups+2)*kv_channels, hidden_size))
    return qkv


def save_checkpoint(queue, args):
    buffered_queue = BufferedQueue(queue)

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import (parse_args, validate_args)
        from megatron.checkpointing import save_checkpoint
        from megatron.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.core import mpu
        from megatron.optimizer import get_param_groups

        from checkpoint_util_megatron import get_models, DummyOptimizer, DummyOptParamScheduler
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)


    md = buffered_queue.get_obj()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1

    if args.target_virtual_pipeline_parallel_size is None:
        if hasattr(md, 'previous_virtual_pipeline_parallel_size'):
            args.target_virtual_pipeline_parallel_size = md.previous_virtual_pipeline_parallel_size
        else:
            print("loader did not provide a virtual pipeline parallel size and --target-virtual-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_virtual_pipeline_parallel_size = 1

    if args.target_expert_parallel_size is None:
        if hasattr(md, 'previous_expert_parallel_size'):
            args.target_expert_parallel_size = md.previous_expert_parallel_size
        else:
            print("loader did not provide a expert model parallel size and --target-expert-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_expert_parallel_size = 1

    if args.target_num_query_groups is None:
        if hasattr(md, 'previous_num_query_groups'):
            args.target_num_query_groups = md.previous_num_query_groups
        else:
            print("loader did not provide num_query_groups and --target-num-query-groups not provided on command line. "
                  "Default to num_attention_heads.")
            args.target_num_query_groups = md.num_attention_heads

    if args.target_moe_grouped_gemm is None:
        if hasattr(md.checkpoint_args, 'moe_grouped_gemm'):
            args.target_moe_grouped_gemm = md.checkpoint_args.moe_grouped_gemm
    else:
        args.target_moe_grouped_gemm = bool(args.target_moe_grouped_gemm)

    if args.target_moe_te_grouped_gemm is None:
        if hasattr(md.checkpoint_args, 'moe_te_grouped_gemm'):
            args.target_moe_te_grouped_gemm = md.checkpoint_args.moe_te_grouped_gemm
    else:
        args.target_moe_te_grouped_gemm = bool(args.target_moe_te_grouped_gemm)

    if args.target_moe_grouped_gemm and args.target_moe_te_grouped_gemm:
        raise ValueError("please set either --target-moe-grouped-gemm or --target-moe-te-grouped-gemm to 0")

    if args.target_vocab_size_divisible_by is None:
        if hasattr(md, 'make_vocab_size_divisible_by'):
            args.target_vocab_size_divisible_by = md.make_vocab_size_divisible_by

    if args.target_kaimm_num_layers_padding_front is None:
        if hasattr(md, 'kaimm_num_layers_padding_front'):
            args.target_kaimm_num_layers_padding_front = md.kaimm_num_layers_padding_front
        else:
            args.target_kaimm_num_layers_padding_front = 0

    if args.target_kaimm_num_layers_padding_back is None:
        if hasattr(md, 'kaimm_num_layers_padding_back'):
            args.target_kaimm_num_layers_padding_back = md.kaimm_num_layers_padding_back
        else:
            args.target_kaimm_num_layers_padding_back = 0

    num_layers_including_padding_layers = (
        md.num_layers +
        args.target_kaimm_num_layers_padding_front +
        args.target_kaimm_num_layers_padding_back
    )
    assert num_layers_including_padding_layers % args.target_pipeline_parallel_size == 0
    num_layers_per_pipeline_stage = num_layers_including_padding_layers // args.target_pipeline_parallel_size

    if args.target_pipeline_parallel_size >= 2:
        assert num_layers_per_pipeline_stage % args.target_virtual_pipeline_parallel_size == 0
        args.num_layers_per_virtual_pipeline_stage = num_layers_per_pipeline_stage // args.target_virtual_pipeline_parallel_size
    else:
        if args.target_virtual_pipeline_parallel_size > 1:
            raise ValueError("Unable to use virtual pipeline parallel, target-pipeline-parallel-size should be greater than or equal to 2.")
        args.num_layers_per_virtual_pipeline_stage = None

    assert args.target_num_query_groups % args.target_tensor_parallel_size == 0, \
        f"Saver set num_query_groups as {args.target_num_query_groups} which is not a multiple of the tp_size {args.target_tensor_parallel_size}"

    print(f"pre_num_query_group: {md.previous_num_query_groups}, tar_num_query_groups: {args.target_num_query_groups}")

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--make-main-grad-addresss-divisible-by', '1',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(
                    args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(
                    args.target_pipeline_parallel_size),
                '--expert-model-parallel-size', str(
                    args.target_expert_parallel_size),
                '--sequence-parallel',
                '--kaimm-num-layers-padding-front', str(
                    args.target_kaimm_num_layers_padding_front),
                '--kaimm-num-layers-padding-back', str(
                    args.target_kaimm_num_layers_padding_back),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--save-interval', '1',
                '--save', args.save_dir
                ]

    if md.tokenizer_model is not None:
        sys.argv.extend(['--tokenizer-model', str(md.tokenizer_model)])
    if md.vocab_size is not None:
        sys.argv.extend(['--vocab-size', str(md.vocab_size)])
    if args.target_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by',
                        str(args.target_vocab_size_divisible_by)])
    if args.num_layers_per_virtual_pipeline_stage is not None:
        sys.argv.extend(['--num-layers-per-virtual-pipeline-stage',
                str(args.num_layers_per_virtual_pipeline_stage)])
    # group query atttentions
    if args.target_num_query_groups != md.num_attention_heads:
        sys.argv.append('--group-query-attention')
        sys.argv.extend(['--num-query-groups', str(args.target_num_query_groups)])
    if args.target_moe_grouped_gemm:
        sys.argv.append('--moe-grouped-gemm')
    if args.target_moe_te_grouped_gemm:
        sys.argv.append('--moe-te-grouped-gemm')

    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.position_embeddings:
        sys.argv.append('--no-position-embedding')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')
    if md.rms_norm:
        sys.argv.append('--rms-norm')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()

    if hasattr(md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size',
                        'expert_model_parallel_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'vocab_size', 'make_vocab_size_divisible_by',
                        'group_query_attention', 'num_query_groups',
                        'moe_grouped_gemm', 'moe_te_grouped_gemm',
                        'kaimm_num_layers_padding_front', 'kaimm_num_layers_padding_back',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay',
                        'global_batch_size', 'world_size', 'context_parallel_size',
                        'overlap_sp_ag', 'overlap_sp_rs',
                        'tensorboard_dir',
                        ]

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(
                    f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(
                    f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    validate_args(margs)

    set_global_variables(margs)

    margs = get_args()
    # model dist
    assert margs.tensor_model_parallel_size == args.target_tensor_parallel_size
    assert margs.pipeline_model_parallel_size == args.target_pipeline_parallel_size
    assert margs.expert_model_parallel_size == args.target_expert_parallel_size
    if args.target_virtual_pipeline_parallel_size > 1:
        assert margs.virtual_pipeline_model_parallel_size == args.target_virtual_pipeline_parallel_size

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'LLAMA':
        from pretrain_llama import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # fake initializing distributed
    print(f"set target tp_size: {args.target_tensor_parallel_size}, pp_size: {args.target_pipeline_parallel_size},"
                    f" vp_size: {args.target_virtual_pipeline_parallel_size}")
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(args.target_virtual_pipeline_parallel_size)
    mpu.set_expert_model_parallel_world_size(args.target_expert_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    # set fake cp for the construction of RotaryEmbedding
    mpu.set_context_parallel_world_size(1)
    mpu.set_context_parallel_rank(0)
    # set fake dp for the checking of saving dataloader states
    mpu.set_data_parallel_world_size(1)
    mpu.set_data_parallel_rank(0)
    # fused_kernels.load(margs)

    save_optimizer = args.process_optimizer
    if save_optimizer:
        # set args to save optimizer state
        # see: megatron/checkpointing.py:save_checkpoint()
        margs.use_distributed_optimizer = True
        margs.no_save_optim = False

    # Prapare models for all pipeline stages
    # all_models[pp_rank][vp_rank][tp_rank] is a model
    # all_optim_models[key][pp_rank][vp_rank][tp_rank] is a model

    def queue_get(name, optional=False, repeat=1):
        meta_info = buffered_queue.get(name, repeat)
        if meta_info is None:
            if not optional:
                print(f'Missing message. Expecting "{name}". Exiting saver.')
                exit(1)
            else:
                print(f"did not receive {name}")
                return None
        print(f"received {name}")

        if "path" in meta_info:
            return torch.load(meta_info["path"])
        else:
            return meta_info

    def queue_get_model(category, name, optional=False, repeat=1):
        real_name = name if category == "model" else f"{category} {name}"
        return queue_get(real_name, optional=optional, repeat=repeat)


    msg = queue_get("done", optional=True)
    if msg != "done":
        print("ERROR: got some more data but was expecting to be done")

    import gc
    # Receive common checkpoint data
    base_ckp = {}
    if save_optimizer:
        base_ckp = queue_get('base_checkpoint')
        
    for pp_rank in range(args.target_pipeline_parallel_size):
        for ep_rank in range(args.target_expert_parallel_size):
            print(f"start saving at pp_rank {pp_rank} ep_rank {ep_rank}")
            mpu.set_expert_model_parallel_rank(ep_rank)
            model = get_models(model_provider, pp_rank, args.target_virtual_pipeline_parallel_size,
                                         args.target_tensor_parallel_size, md.params_dtype)
            receive_and_split_models(model, pp_rank, args.target_expert_parallel_size, md, margs, args,
                                        partial(queue_get_model, "model"))
            print("finish receive_and_split_models")  

            if save_optimizer:
                all_optim_models = {}
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    all_optim_models[key] = []
                for key, value in all_optim_models.items():
                    all_optim_models[key] = get_models(model_provider, pp_rank, args.target_virtual_pipeline_parallel_size,
                                            args.target_tensor_parallel_size, torch.float32)
                    receive_and_split_models(all_optim_models[key], pp_rank, args.target_expert_parallel_size, md, margs, args,
                                     partial(queue_get_model, f"optim-{key}"))
                    print(f"finish saving all_optim_models[{key}]")

            mpu.set_pipeline_model_parallel_rank(pp_rank)
            for tp_rank in range(args.target_tensor_parallel_size):
                mpu.set_tensor_model_parallel_rank(tp_rank)
                models = [model[vp_rank][tp_rank]
                          for vp_rank in range(args.target_virtual_pipeline_parallel_size)]
                optimizer = None
                opt_param_scheduler = None
                if save_optimizer:
                    optim_models = {}
                    for (key, value) in all_optim_models.items():
                        optim_models[key] = [value[vp_rank][tp_rank]
                                             for vp_rank in range(len(value))]
                    model_chunks_for_param_groups = [m[tp_rank] for m in model]
                    # change here if no_weight_decay_cond or scale_lr_cond is specified
                    param_groups = get_param_groups(model_chunks_for_param_groups, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0)
                    del model_chunks_for_param_groups
                    optimizer_state_dict = DummyOptimizer.migrate_state_dict_on_demand(base_ckp['optimizer'], param_groups, args.target_expert_parallel_size)
                    optimizer = DummyOptimizer(
                        optim_models, state_dict=optimizer_state_dict)
                    opt_param_scheduler = DummyOptParamScheduler(
                        base_ckp['opt_param_scheduler'])
                print(f"save_checkpoint pp_rank {pp_rank} tp_rank {tp_rank}")
                save_checkpoint(md.iteration, models, optimizer=optimizer,
                                opt_param_scheduler=opt_param_scheduler, dataloader=None)
                del models
                if save_optimizer:
                    del optim_models
                gc.collect()
                print(f"del models at pp_rank {pp_rank} ep_rank {ep_rank} tp_rank {tp_rank}")

            del model
            if save_optimizer:
                del all_optim_models
            gc.collect()
            print(f"del optimizers at pp_rank {pp_rank} ep_rank {ep_rank}")
    print("Complete convertion")


def receive_and_split_models(pp_model, pp_rank, ep_size, md, margs, args, queue_get):
    """
        receive parameters from loader and split them by tp dimension, then save them to models
    """
    from megatron.core import mpu
    from megatron.core.transformer.moe.moe_layer import BaseMoELayer
    from megatron.model.transformer import NoopTransformerLayer

    if(pp_rank == 0):
        # Embeddings
        embeddings_msg = queue_get("embeddings", repeat=ep_size)

        if md.position_embeddings:
            pos_embed = embeddings_msg["position embeddings"]
        orig_word_embed = embeddings_msg["word embeddings"]
        print(f"orig_word_embed.shape = {orig_word_embed.shape}")

        # Deal with padding
        if margs.padded_vocab_size is not None:
            print(f"margs.padded_vocab_size = {margs.padded_vocab_size}")
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            print(f"orig_vocab_size = {orig_vocab_size}")

            # Cut out extra padding we don't need
            if orig_vocab_size > margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0:margs.padded_vocab_size, :]

            # Expanding embedding to larger size by replicating final entry
            elif orig_vocab_size < margs.padded_vocab_size:
                padding_size = margs.padded_vocab_size - orig_vocab_size
                padding_tensor = torch.zeros(padding_size, orig_word_embed.size(1), dtype=orig_word_embed.dtype, device=orig_word_embed.device)
                full_word_embed = torch.cat((orig_word_embed, padding_tensor), dim=0)
                # full_word_embed = torch.cat((
                #     orig_word_embed,
                #     orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))
                print(f"full_word_embed.shape = {full_word_embed.shape}")
            else:
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems.")
            margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed

        # Split into new tensor model parallel sizes
        out_word_embed = torch.chunk(
            full_word_embed, args.target_tensor_parallel_size, dim=0)
        for i, chunk in enumerate(out_word_embed):
            print(f"out_word_embed[{i}].shape = {chunk.shape}")
        # Make models for first pipeline stage and fill in embeddings
        mpu.set_pipeline_model_parallel_rank(0)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
        models = pp_model[0]
        for tp_rank, model in enumerate(models):
            model.language_model.embedding.word_embeddings.weight.data.copy_(
                out_word_embed[tp_rank])
            if md.position_embeddings:
                model.language_model.embedding.position_embeddings.weight.data.copy_(
                    pos_embed)
            else:
                assert not hasattr(model.language_model.embedding,
                                "position_embeddings")

    # Transformer layers
    total_layer_num = 0
    # OOM: 一次性从队列中得到完整的模型导致 OOM, 应当只得到当前需要的模型
    for vp_rank in range(args.target_virtual_pipeline_parallel_size):
        # Get the models for this pipeline stage
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        models = pp_model[vp_rank]
        num_layers = len(models[0].language_model.encoder.layers)
        for layer_num in range(num_layers):
            layer_tp_rank0 = models[0].language_model.encoder.layers[layer_num]
            if isinstance(layer_tp_rank0, NoopTransformerLayer):
                continue
            total_layer_num = layer_tp_rank0.layer_number - 1
            print(f"vp_rank: {vp_rank}, args.target_virtual_pipeline_parallel_size: {args.target_virtual_pipeline_parallel_size},"
                    f" args.target_pipeline_parallel_size: {args.target_pipeline_parallel_size}, num_layers: {num_layers},"
                    f" pp_rank: {pp_rank}, layer_num: {layer_num}, total_layer_num: {total_layer_num}")
            msg = queue_get(f"transformer layer {total_layer_num}", repeat=ep_size)

            # duplicated tensors
            is_moe_layer = isinstance(layer_tp_rank0.mlp, BaseMoELayer)
            input_layernorm_weight = msg["input layernorm weight"]
            input_layernorm_bias = msg["input layernorm bias"]
            post_layernorm_weight = msg["post layernorm weight"]
            post_layernorm_bias = msg["post layernorm bias"]
            if md.linear_bias:
                dense_bias = msg["dense bias"]
                mlp_l1_bias = msg["mlp l1 bias"]
            if is_moe_layer:
                router_weight = msg["router weight"]
                if md.checkpoint_args.shared_expert_hidden_size and md.checkpoint_args.shared_expert_combine_method == "softmax":
                    moe_coefficient_weight = msg["moe coefficient weight"]
                    moe_coefficient_bias = msg["moe coefficient bias"]

            # Split up the parallel tensors
            tar_qkv_weights = _megatron_convert_group_query_attentions(msg["qkv weight"], 
                md.previous_num_query_groups, args.target_num_query_groups, md.num_attention_heads, md.hidden_size//md.num_attention_heads)
            qkv_weight = torch.chunk(
                tar_qkv_weights, args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(
                msg["dense weight"], args.target_tensor_parallel_size, dim=1)
            if md.linear_bias:
                qkv_bias = torch.chunk(
                    msg["qkv bias"], args.target_tensor_parallel_size, dim=0)

            def msg_to_mlp_weight(msg):
                mlp_l1_weight = torch.chunk(
                    msg["mlp l1 weight"], args.target_tensor_parallel_size, dim=1)

                # Special handling for swiglu
                if md.swiglu:
                    mlp_l0_weight_W = torch.chunk(
                        msg["mlp l0 weight W"], args.target_tensor_parallel_size, dim=0)
                    mlp_l0_weight_V = torch.chunk(
                        msg["mlp l0 weight V"], args.target_tensor_parallel_size, dim=0)
                    mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(
                        mlp_l0_weight_W, mlp_l0_weight_V)]
                else:
                    mlp_l0_weight = torch.chunk(
                        msg["mlp l0 weight"], args.target_tensor_parallel_size, dim=0)

                if md.linear_bias:
                    if md.swiglu:
                        mlp_l0_bias_W = torch.chunk(
                            msg["mlp l0 bias W"], args.target_tensor_parallel_size, dim=0)
                        mlp_l0_bias_V = torch.chunk(
                            msg["mlp l0 bias V"], args.target_tensor_parallel_size, dim=0)
                        mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(
                            mlp_l0_bias_W, mlp_l0_bias_V)]
                    else:
                        mlp_l0_bias = torch.chunk(
                            msg["mlp l0 bias"], args.target_tensor_parallel_size, dim=0)
                else:
                    mlp_l0_bias = []

                return mlp_l0_weight, mlp_l0_bias, mlp_l1_weight

            if is_moe_layer:
                experts_weight = dict()
                for local_expert_index in layer_tp_rank0.mlp.local_expert_indices:
                    expert_msg = queue_get(f"transformer layer {total_layer_num} expert {local_expert_index}")
                    print("queue_get", f"transformer layer {total_layer_num} expert {local_expert_index}")
                    mlp_l0_weight, mlp_l0_bias, mlp_l1_weight = msg_to_mlp_weight(expert_msg)
                    experts_weight[local_expert_index] = {
                        "mlp_l0_weight": mlp_l0_weight,
                        "mlp_l0_bias": mlp_l0_bias,
                        "mlp_l1_weight": mlp_l1_weight,
                    }
                if md.checkpoint_args.shared_expert_hidden_size:
                    expert_msg_prefix = f"moe shared_expert "
                    expert_msg = {
                        k[len(expert_msg_prefix):]: v
                        for k, v in msg.items()
                        if k.startswith(expert_msg_prefix)
                    }
                    mlp_l0_weight, mlp_l0_bias, mlp_l1_weight = msg_to_mlp_weight(expert_msg)
            else:
                mlp_l0_weight, mlp_l0_bias, mlp_l1_weight = msg_to_mlp_weight(msg)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].language_model.encoder.layers[layer_num]
                l.input_layernorm.weight.data.copy_(input_layernorm_weight)
                if hasattr(l.input_layernorm, "bias"):
                    l.input_layernorm.bias.data.copy_(input_layernorm_bias)
                l.self_attention.query_key_value.weight.data.copy_(
                    qkv_weight[tp_rank])
                l.self_attention.dense.weight.data.copy_(
                    dense_weight[tp_rank])
                l.post_attention_layernorm.weight.data.copy_(
                    post_layernorm_weight)
                if hasattr(l.post_attention_layernorm, "bias"):
                    l.post_attention_layernorm.bias.data.copy_(
                        post_layernorm_bias)
                if md.linear_bias:
                    l.self_attention.query_key_value.bias.data.copy_(
                        qkv_bias[tp_rank])
                    l.self_attention.dense.bias.data.copy_(dense_bias)
                    l.mlp.dense_4h_to_h.bias.data.copy_(mlp_l1_bias)

                def copy_mlp_weight(mlp, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight):
                    mlp.dense_h_to_4h.weight.data.copy_(
                        mlp_l0_weight[tp_rank])
                    mlp.dense_4h_to_h.weight.data.copy_(
                        mlp_l1_weight[tp_rank])
                    if md.linear_bias:
                        mlp.dense_h_to_4h.bias.data.copy_(
                            mlp_l0_bias[tp_rank])

                if is_moe_layer:
                    l.mlp.router.weight.data.copy_(router_weight)
                    if md.checkpoint_args.shared_expert_hidden_size and md.checkpoint_args.shared_expert_combine_method == "softmax":
                        l.mlp.coefficient.weight.data.copy_(moe_coefficient_weight)
                        l.mlp.coefficient.bias.data.copy_(moe_coefficient_bias)
                    if margs.moe_grouped_gemm:
                        if md.linear_bias:
                            raise NotImplementedError("bias in the expert layer is not supported in Grouped GEMM yet")
                        w1 = l.mlp.experts.weight1.view(l.mlp.experts.num_local_experts, l.mlp.experts.config.hidden_size, -1)
                        w2 = l.mlp.experts.weight2.view(l.mlp.experts.num_local_experts, -1, l.mlp.experts.config.hidden_size)
                        for local_expert_index, mlp_l0_weight_T, mlp_l1_weight_T in zip(l.mlp.local_expert_indices, w1, w2):
                            mlp_l0_weight_T.data.copy_(experts_weight[local_expert_index]["mlp_l0_weight"][tp_rank].T)
                            mlp_l1_weight_T.data.copy_(experts_weight[local_expert_index]["mlp_l1_weight"][tp_rank].T)
                    elif margs.moe_te_grouped_gemm:
                        if md.linear_bias:
                            raise NotImplementedError("bias in the expert layer is not supported in Grouped GEMM yet")
                        for i, local_expert_index in enumerate(l.mlp.local_expert_indices):
                            getattr(l.mlp.experts.linear_fc1, f"weight{i}").data.copy_(experts_weight[local_expert_index]["mlp_l0_weight"][tp_rank])
                            getattr(l.mlp.experts.linear_fc2, f"weight{i}").data.copy_(experts_weight[local_expert_index]["mlp_l1_weight"][tp_rank])
                    else:
                        for i, local_expert_index in enumerate(l.mlp.local_expert_indices):
                            copy_mlp_weight(l.mlp.experts.local_experts[i],
                                            experts_weight[local_expert_index]["mlp_l0_weight"],
                                            experts_weight[local_expert_index]["mlp_l0_bias"],
                                            experts_weight[local_expert_index]["mlp_l1_weight"])
                    if md.checkpoint_args.shared_expert_hidden_size:
                        copy_mlp_weight(l.mlp.shared_expert, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight)
                else:
                    copy_mlp_weight(l.mlp, mlp_l0_weight, mlp_l0_bias, mlp_l1_weight)

        post_process = mpu.is_pipeline_last_stage()
        if post_process:
            msg = queue_get("final layernorm", repeat=ep_size)
            final_layernorm_weight = msg["weight"]
            final_layernorm_bias = msg["bias"]
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].language_model.encoder.final_layernorm.weight.data.copy_(
                    final_layernorm_weight)
                if hasattr(models[tp_rank].language_model.encoder.final_layernorm, "bias"):
                    models[tp_rank].language_model.encoder.final_layernorm.bias.data.copy_(
                        final_layernorm_bias)
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].word_embeddings.weight.data.copy_(
                        out_word_embed[tp_rank])
            del final_layernorm_weight
            del final_layernorm_bias

            if md.output_layer:
                msg = queue_get("output layer", repeat=ep_size)
                if not hasattr(models[0].language_model, 'output_layer'):
                    print(
                        "ERROR: got an output layer, but model does not have one")
                    exit(1)
                print(f"old output weight.shape = {msg['weight'].shape}")

                # Deal with padding
                if margs.padded_vocab_size is not None:
                    orig_vocab_size = msg['weight'].shape[0]
                    # Cut out extra padding we don't need
                    if orig_vocab_size > margs.padded_vocab_size:
                        new_output_weight = msg['weight'][0:margs.padded_vocab_size, :]
                    # Expanding embedding to larger size by replicating final entry
                    elif orig_vocab_size < margs.padded_vocab_size:
                        padding_size = margs.padded_vocab_size - orig_vocab_size
                        padding_tensor = torch.zeros(padding_size, msg["weight"].size(1), dtype=msg["weight"].dtype, device=msg["weight"].device)
                        new_output_weight = torch.cat((msg['weight'], padding_tensor), dim=0)
                    else:
                        new_output_weight = msg['weight']
                else:
                    print("Original vocab size not specified, leaving embedding table as-is. "
                        "If you've changed the tensor parallel size this could cause problems.")
                    margs.padded_vocab_size = msg['weight'].shape[0]
                    new_output_weight = msg['weight']

                output_layer_weight = torch.chunk(
                    new_output_weight, args.target_tensor_parallel_size, dim=0)
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].language_model.output_layer.weight.data.copy_(
                        output_layer_weight[tp_rank])
                    print(f"tp_rank: {tp_rank}, model_output_layer.shape: {models[tp_rank].language_model.output_layer.weight.data.shape},"
                            f" msg_output_layer_weight.shape: {output_layer_weight[tp_rank].shape}")
                del output_layer_weight

            msg = queue_get("poller", optional=True, repeat=ep_size)
            if msg is not None:
                if not hasattr(models[0].language_model, 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg["weight"]
                pooler_bias = msg["bias"]
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].language_model.pooler.dense.weight.data.copy_(
                        pooler_weight)
                    models[tp_rank].language_model.pooler.dense.bias.data.copy_(
                        pooler_bias)
                del pooler_weight
                del pooler_bias

            msg = queue_get("lm head", optional=True, repeat=ep_size)
            if msg is not None:
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg["dense weight"]
                lm_head_dense_bias = msg["dense bias"]
                lm_head_layernorm_weight = msg["layernorm weight"]
                lm_head_layernorm_bias = msg["layernorm bias"]
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(
                        lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(
                        lm_head_dense_bias)
                    models[tp_rank].lm_head.layernorm.weight.data.copy_(
                        lm_head_layernorm_weight)
                    models[tp_rank].lm_head.layernorm.bias.data.copy_(
                        lm_head_layernorm_bias)

            msg = queue_get("binary head", optional=True, repeat=ep_size)
            if msg is not None:
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg["weight"]
                binary_head_bias = msg["bias"]
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(
                        binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(
                        binary_head_bias)
