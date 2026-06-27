# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.profile_utils import annotate_forward_backward
from torch.utils.checkpoint import detach_variable

from . import LayerNorm, RMSNorm
from .enums import AttnMaskType, LayerType
from .module import MegatronModule
from .rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
from .transformer import ParallelTransformer
from .utils import get_linear_layer
from .utils import init_method_normal, init_method_xavier_normal
from .utils import scaled_init_method_normal, scaled_init_method_xavier_normal


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None, pipeline_parallel=False):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel or pipeline_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel and not pipeline_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel_enabled=args.sequence_parallel or pipeline_parallel,
        pipeline_parallel=pipeline_parallel)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(num_tokentypes, add_pooler,
                       encoder_attn_mask_type, init_method=None,
                       scaled_init_method=None, add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        if args.init_method_xavier_normal:
            init_method = init_method_xavier_normal(args.init_method_beta)
        else:
            init_method = init_method_normal(args.init_method_std)


    if scaled_init_method is None:
        if args.init_method_xavier_normal:
            scaled_init_method = scaled_init_method_xavier_normal(args.init_method_beta,
                                                       args.num_layers)
        else:
            scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                        args.num_layers)


    # Language model.
    language_model = TransformerLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = args.sequence_parallel


    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        embedding_weights_in_fp32: casts word embedding weights to
                                   fp32 before sampling. Required to
                                   maintain reproducibility when
                                   training in bf16.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0,
                 embedding_weights_in_fp32=False,
                 vocab_in_pp=False):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method,
            params_dtype=args.params_dtype,
            use_cpu_initialization=args.use_cpu_initialization,
            perform_initialization=args.perform_initialization,
            padding_idx=args.pad_token_id,
            vocab_in_pp=vocab_in_pp
        )
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.add_position_embedding = args.add_position_embedding
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            if args.perform_initialization:
                self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    @annotate_forward_backward("emb", "emb")
    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        # [b s] --> [s b].
        input_ids = input_ids.transpose(0, 1)
        words_embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)
        if self.add_position_embedding:
            # [b s] --> [s b].
            position_ids = position_ids.transpose(0, 1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # input_ids already transposed before.
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(prefix=prefix,
                                              keep_vars=keep_vars)
        if self.add_position_embedding:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(prefix=prefix,
                                                       keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.add_position_embedding:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        # TODO: passing share_word_embeddings=False will not work correctly for T5 and embeddings will not be synced. Fix later for T5.
        if args.untie_embeddings_and_output_weights: assert not add_decoder
        super(TransformerLanguageModel, self).__init__(share_word_embeddings=not args.untie_embeddings_and_output_weights)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.add_retriever = args.retro_add_retriever
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.vocab_in_pp = args.kaimm_vocab_in_pipeline_parallel
        self.first_vstage = not mpu.get_virtual_pipeline_model_parallel_rank()  # None or 0

        # Embeddings.
        if not self.vocab_in_pp and self.pre_process or self.vocab_in_pp and self.first_vstage:
            # Judge if use xavier normal. Judge if set emb std when use normal.
            if args.init_method_xavier_normal:
                init_method = self.init_method
            elif args.init_method_std_emb is None:
                init_method = self.init_method
            else:
                init_method = init_method_normal(args.init_method_std_emb)


            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       init_method,
                                       self.num_tokentypes,
                                       args.embedding_weights_in_fp32,
                                       self.vocab_in_pp)
            self._embedding_key = 'embedding'

        # Final layer norm before output.
        if self.vocab_in_pp and self.first_vstage:
            if args.rms_norm:
                self.final_layernorm = RMSNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    sequence_parallel=False,
                    no_persist_rms_norm = not args.use_fast_rms_norm,
                    apply_layernorm_1p=args.apply_layernorm_1p,
                    memory_efficient=args.use_memory_efficient_norm)
            else:
                self.final_layernorm = LayerNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=False,
                    apply_layernorm_1p=args.apply_layernorm_1p,
                    memory_efficient=args.use_memory_efficient_norm)
            # The weight is replicated across PP, and only one's grad should be counted for grad_norm.
            if not mpu.is_pipeline_first_stage(ignore_virtual=True):
                for param in self.final_layernorm.parameters():
                    param.shared = True

        # Rotary positional embeddings
        self.use_rotary_position_embeddings = \
            args.use_rotary_position_embeddings
        if args.use_rotary_position_embeddings:
            self.seq_length = args.micro_seq_length or args.seq_length
            rotary_dim = args.hidden_size // args.num_attention_heads \
                if args.kv_channels is None else args.kv_channels

            if args.rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * args.rotary_percent)

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim, args.rope_theta, args.use_fast_rope,
                                                  mpu.get_context_parallel_world_size(),
                                                  mpu.get_context_parallel_rank())

        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                model_type=args.model_type if not args.retro_add_retriever \
                    else ModelType.retro_decoder,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process and not self.vocab_in_pp,
                post_process=self.post_process and not self.vocab_in_pp,
            )
            self._encoder_key = 'encoder'
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                model_type=args.model_type,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process)
            self._decoder_key = 'decoder'
        else:
            self.decoder = None

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

            if self.untie_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    args.hidden_size,
                    args.padded_vocab_size,
                    bias=False, # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                    init_method=self.init_method,
                    use_cpu_initialization=args.use_cpu_initialization,
                    perform_initialization=args.perform_initialization)
                self._output_layer_key = 'output_layer'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask, kv_cache,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):

        # Encoder embedding.
        if not self.vocab_in_pp and self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # Retriever embedding.
        if self.add_retriever and self.pre_process:
            retriever_input = self.embedding(retriever_input_ids,
                                             retriever_position_ids,
                                             tokentype_ids=tokentype_ids)
        else:
            retriever_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                rotary_pos_emb = \
                    self.rotary_pos_emb(inference_params.max_sequence_len)
            else:
                if isinstance(enc_position_ids, int):
                    rotary_pos_emb = self.rotary_pos_emb(self.seq_length, enc_position_ids)
                else:
                    rotary_pos_emb = self.rotary_pos_emb(enc_position_ids)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    kv_cache=kv_cache,
                    retriever_input=retriever_input,
                    retriever_attn_mask=retriever_attn_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                keep_vars=keep_vars)
        if self.add_encoder:
            state_dict_[self._encoder_key] \
                = self.encoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process or self.vocab_in_pp:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
            if self.untie_embeddings_and_output_weights:
                output_layer = self.output_layer.state_dict(prefix=prefix, keep_vars=keep_vars)
                if self.vocab_in_pp:
                    weight = output_layer['weight']
                    if mpu.is_pipeline_first_stage(ignore_virtual=True):
                        gather_list = [torch.empty_like(weight)
                                       for _ in range(mpu.get_pipeline_model_parallel_world_size())]
                    else:
                        gather_list = None
                    dst = mpu.get_pipeline_model_parallel_first_rank()
                    torch.distributed.gather(weight, gather_list, dst, group=mpu.get_pipeline_model_parallel_group())
                    if mpu.is_pipeline_first_stage(ignore_virtual=True):
                        output_layer['weight'] = torch.cat(gather_list)
                        state_dict_[self._output_layer_key] = output_layer
                else:
                    state_dict_[self._output_layer_key] = output_layer

        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if '.attention.' in key:
                    state_dict_self_attention[key.replace(".attention.",
                        ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process or self.vocab_in_pp:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
            if self.untie_embeddings_and_output_weights:
                if self.vocab_in_pp:
                    if mpu.is_pipeline_first_stage(ignore_virtual=True):
                        output_layer = state_dict[self._output_layer_key]
                        weight = output_layer['weight'].cuda()
                        scatter_list = list(weight.chunk(mpu.get_pipeline_model_parallel_world_size()))
                    else:
                        output_layer = self.output_layer.state_dict()
                        scatter_list = None
                    weight = torch.empty_like(self.output_layer.weight)
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    torch.distributed.scatter(weight, scatter_list, src, group=mpu.get_pipeline_model_parallel_group())
                    output_layer['weight'] = weight
                    state_dict[self._output_layer_key] = output_layer
                assert 'output_layer' in state_dict, \
                    'could not find data for output_layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key],
                                                  strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)


class _SlicePostLMProcessingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, slice_size, lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy):
        args = get_args()
        SP = args.tensor_model_parallel_size if args.sequence_parallel else 1
        ctx.SP = SP
        assert slice_size % SP == 0
        slice_size_by_SP = slice_size // SP

        ctx.run_function = run_function
        ctx.slice_size_by_SP = slice_size_by_SP
        ctx.parallel_output = parallel_output
        ctx.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        ctx.main_grad = logit_weights.main_grad

        s_by_SP, b = lm_output.shape[:2]
        losses = []
        for i in range(0, s_by_SP, slice_size_by_SP):
            with torch.no_grad():
                labels_i = labels.view(b, SP, -1)[:, :, i:i + slice_size_by_SP].reshape(b, -1)
                loss_i = run_function(lm_output[i:i + slice_size_by_SP], labels_i, logit_weights, parallel_output, fp16_lm_cross_entropy)
                losses.append(loss_i.view(b, SP, -1))
        loss = torch.cat(losses, dim=2).view(b, -1)  # (b, s)
        ctx.save_for_backward(lm_output, labels, logit_weights)

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        lm_output, labels, logit_weights = ctx.saved_tensors

        # Compute the forward pass.
        lm_output, labels, logit_weights = detach_variable((lm_output, labels, logit_weights))
        logit_weights.main_grad = ctx.main_grad

        s_by_SP, b = lm_output.shape[:2]
        slice_size_by_SP = ctx.slice_size_by_SP
        SP = ctx.SP
        for i in range(0, s_by_SP, slice_size_by_SP):
            labels_i = labels.view(b, SP, -1)[:, :, i:i + slice_size_by_SP].reshape(b, -1)
            grad_loss_i = grad_loss.view(b, SP, -1)[:, :, i:i + slice_size_by_SP].reshape(b, -1)
            with torch.enable_grad():
                loss_i = ctx.run_function(lm_output[i:i + slice_size_by_SP], labels_i, logit_weights, ctx.parallel_output, ctx.fp16_lm_cross_entropy)
            torch.autograd.backward(loss_i, grad_loss_i)

        return None, None, lm_output.grad, labels.grad, logit_weights.grad, None, None


def slice_post_lm_processing(run_function, slice_size, lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy):
    return _SlicePostLMProcessingFunction.apply(run_function, slice_size, lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy)
