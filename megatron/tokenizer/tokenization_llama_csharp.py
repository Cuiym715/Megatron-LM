# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Collection, Dict, List, Set, Tuple, Union
import threading

"""Tokenization classes for LLaMA."""
import os
import sys
import json
from sys import *
import subprocess
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import sentencepiece as spm


from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "kwaii_tokenizer"}

# PRETRAINED_VOCAB_FILES_MAP = {
#     "vocab_file": {
#         "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
#     },
#     "tokenizer_file": {
#         "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
#     },
# }
# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
#     "hf-internal-testing/llama-tokenizer": 2048,
# }


class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
#     pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
#     max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        unk_token="<unk>"
        bos_token="<s>"
        eos_token="</s>"
        pad_token="<pad>"
        self.token2id = {}
        self.id2token = {}
        #bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        #eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        #unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        #pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        stderr.write(vocab_file +  '!!!!!!!!\n')
        path = '/'.join(vocab_file.split('/')[:-1])
        self.encode_lock = threading.Lock()
        self.encoder = subprocess.Popen([path + '/cstokenizer', '--encode', vocab_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.decoder = subprocess.Popen([path + '/cstokenizer', '--decode', vocab_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.errors = 'replace'
        with open(vocab_file, encoding='utf-8-sig') as f:
            self.json_dict = json.load(f)
        self.id2token = {}
        self.token2id = {}
        self.special_token2id = {}
        self.special_id2token = {}
        for one in self.json_dict["CommonTokens"]:
            self.id2token[one['TokenID']] = bytes(one['TokenBytes'])
            self.token2id[bytes(one['TokenBytes'])] = one['TokenID']
        for one in self.json_dict["SpecialTokens"]:
            self.id2token[one['TokenID']] = one['TokenStr'].encode('utf8')
            self.special_token2id[one['TokenStr']] = one['TokenID']
            self.token2id[one['TokenStr'].encode('utf8')] = one['TokenID']
            self.special_id2token[one['TokenID']] = one['TokenStr']

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state
    def __del__(self):
        pass
        # 在对象销毁前等待10秒
#         print("Tokenizer销毁前等待10秒...")
#         print("Tokenizer已销毁。")
    def __setstate__(self, d):
        self.__dict__ = d
        self.encoder = subprocess.Popen([path + '/cstokenizer', '--encode', vocab_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.errors = 'replace'
        with open(self.vocab_file, encoding='utf-8-sig') as f:
            self.json_dict = json.load(f)
        self.id2token = {}
        self.token2id = {}
        for one in self.json_dict["CommonTokens"]:
            self.id2token[one['TokenID']] = one['TokenStr'].encode('utf8')
            self.token2id[bytes(one['TokenBytes'])] = one['TokenID']
        for one in self.json_dict["SpecialTokens"]:
            self.id2token[one['TokenID']] = one['TokenStr'].encode('utf8')
            self.token2id[bytes(one['TokenBytes'])] = one['TokenID']

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return len(self.id2token)

    def get_vocab(self):
        """Returns vocab as a dict"""
        return self.token2id
    @property
    def bos_token_id(self):
        return self.special_token2id[self.bos_token]
    @property
    def eos_token_id(self):
        return self.special_token2id[self.eos_token]
    @property
    def pad_token_id(self):
        return self.special_token2id[self.pad_token]
    def tokenize(
        self,
        text: str,
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """Returns a tokenized string."""
        j = {}
        j['text'] = text
        s = str(json.dumps(j, ensure_ascii = False))
        res = ''
        with self.encode_lock:
            self.encoder.stdin.write(s + "\n")
            self.encoder.stdin.flush()  # Important to ensure data is actually sent
            res = '[' + self.encoder.stdout.readline().strip() + ']'
        segs = json.loads(res)
        tokens = [self.id2token[one] for one in segs]
        if self.add_bos_token:
            tokens = [self.bos_token] + tokens
        if self.add_eos_token:
            tokens = tokens + [self.eos_token]
            
          
#         print(self.add_bos_token)
#         print(self.add_eos_token)
        
        return tokens
        

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_token2id:
            return self.special_token2id[token]
        return self.token2id[token]

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.id2token[index]
        return token

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            spaces_between_special_tokens: bool = True,
            **kwargs,
    ) -> str:
        if skip_special_tokens:
            token_ids = filter(lambda x: x not in self.special_id2token, token_ids)
        s = ','.join([str(x) for x in token_ids])
        self.decoder.stdin.write(s + "\n")
        self.decoder.stdin.flush()  # Important to ensure data is actually sent
        j = json.loads(self.decoder.stdout.readline().strip())
        text = j['text']

        return text

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ''
