# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""llama model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.profile_utils import annotate_forward_backward
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits, slice_post_lm_processing
from .language_model import get_language_model
from .utils import init_method_xavier_normal, init_method_normal
from .utils import scaled_init_method_xavier_normal, scaled_init_method_normal
from .utils import slice_lm_inputs_along_cp, gather_post_lm_output_along_cp
from .utils import pad_and_permute, unpermute_and_unpad


@annotate_forward_backward("post", "post")
def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy,
                                   pipeline_parallel):
    args = get_args()
    if args.kaimm_post_lm_processing_slice_size:
        if pipeline_parallel:
            raise NotImplementedError()
        return slice_post_lm_processing(post_language_model_processing_inner,
                                        args.kaimm_post_lm_processing_slice_size // args.context_parallel_size,
                                        lm_output, labels, logit_weights,
                                        parallel_output,
                                        fp16_lm_cross_entropy)
    else:
        return post_language_model_processing_inner(lm_output, labels, logit_weights,
                                                    parallel_output,
                                                    fp16_lm_cross_entropy,
                                                    pipeline_parallel)


def post_language_model_processing_inner(lm_output, labels, logit_weights,
                                         parallel_output,
                                         fp16_lm_cross_entropy,
                                         pipeline_parallel):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output,
        pipeline_parallel=pipeline_parallel)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if pipeline_parallel:
            # XXX(lizhouyang): The `output` is transposed at seq dim in [TP, PP] -> [PP, TP],
            #                  by the all_gather in `parallel_lm_logits`.
            #                  Rather than transpose the `output` back, we transpose the `labels` as the `output`.
            #                  And then we transpose the `loss` back at the end.
            labels_shape = labels.shape
            labels = pad_and_permute(labels, 0)
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels,
                                                                pipeline_parallel=pipeline_parallel)
        else:
            output = output.float()
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels,
                                                                pipeline_parallel=pipeline_parallel)

        if pipeline_parallel:
            loss = unpermute_and_unpad(loss, 0, labels_shape)
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class LlamaModel(MegatronModule):
    """llama-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super(LlamaModel, self).__init__(share_word_embeddings=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.vocab_in_pp = args.kaimm_vocab_in_pipeline_parallel

        if args.init_method_xavier_normal:
            init_method =  init_method_xavier_normal(args.init_method_beta)
            scaled_init_method = scaled_init_method_xavier_normal(args.init_method_beta,
                                                         args.num_layers)
        else:
            init_method = init_method_normal(args.init_method_std)
            scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method,
            scaled_init_method=scaled_init_method,

            pre_process=self.pre_process,
            post_process=self.post_process)

        if not args.kaimm_vocab_in_pipeline_parallel and not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings(init_method)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, kv_cache,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):

        if not (self.pre_process and self.vocab_in_pp):
            input_ids, position_ids, attention_mask, labels = \
                slice_lm_inputs_along_cp(input_ids, position_ids, attention_mask, labels)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            kv_cache=kv_cache,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            if self.vocab_in_pp:
                lm_output._labels = labels
                return lm_output
            else:
                return self.post_process_forward(lm_output, labels)
        else:
            return lm_output

    def pre_process_forward(self, input_ids):
        return self.language_model.embedding(input_ids, None)

    def post_process_forward(self, lm_output, labels, pipeline_parallel=False):
        if self.vocab_in_pp:
            lm_output = self.language_model.final_layernorm(lm_output)
        if self.untie_embeddings_and_output_weights:
            weight = self.language_model.output_layer.weight
        else:
            weight = self.language_model.embedding.word_embeddings.weight \
                if self.vocab_in_pp else self.word_embeddings_weight()
        return gather_post_lm_output_along_cp(post_language_model_processing(
            lm_output, labels, weight,
            self.parallel_output,
            self.fp16_lm_cross_entropy,
            pipeline_parallel))

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
