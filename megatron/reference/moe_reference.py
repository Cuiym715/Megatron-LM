import contextlib
import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama import configuration_llama, modeling_llama


_LlamaMLP = None
_LAYER_NUMBER = None
_AUX_LOSS_STORE = None


@contextlib.contextmanager
def rewrite_config(config, name, value):
    config_exists = hasattr(config, name)
    if config_exists:
        value_orig = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        if config_exists:
            setattr(config, name, value_orig)
        else:
            delattr(config, name)


class LlamaMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.moe_router_topk
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = getattr(config, "expert_hidden_size", None) or config.intermediate_size
        self.shared_expert_hidden_size = getattr(config, "shared_expert_hidden_size", None)
        self.shared_expert_combine_method = getattr(config, "shared_expert_combine_method", None)
        self.moe_expert_capacity_factor = getattr(config, "moe_expert_capacity_factor", None)
        self.moe_token_drop_policy = getattr(config, "moe_token_drop_policy", None)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        with rewrite_config(config, "intermediate_size", self.expert_hidden_size):
            self.experts = torch.nn.ModuleList(_LlamaMLP(config) for _ in range(self.num_experts))
        if self.shared_expert_hidden_size:
            with rewrite_config(config, "intermediate_size", self.shared_expert_hidden_size):
                self.shared_expert = _LlamaMLP(config)
            if self.shared_expert_combine_method == "softmax":
                self.coefficient = nn.Linear(self.hidden_size, 2)

    def forward(self, hidden_states):
        identity = hidden_states
        logits = self.router(hidden_states)
        expert_indices = logits.topk(self.topk).indices
        S = hidden_states.shape[:2].numel()
        if _AUX_LOSS_STORE is not None:
            c_e = F.one_hot(expert_indices.view(-1), num_classes=self.num_experts).sum(0).float() / self.topk
            m_e = logits.view(-1, self.num_experts).softmax(dim=1, dtype=torch.float32).sum(0) / S
            aux_loss = ((c_e / S) * m_e).sum() * self.num_experts
            _AUX_LOSS_STORE.append(aux_loss)
        mask = torch.scatter(torch.full_like(logits, -torch.inf), -1, expert_indices, torch.zeros_like(logits))
        logits = logits + mask
        gate = logits.softmax(dim=-1, dtype=torch.float32).type_as(logits)
        if self.training and self.moe_expert_capacity_factor is not None:
            expert_capacity = math.ceil(S * self.topk / self.num_experts * self.moe_expert_capacity_factor)
            if self.moe_token_drop_policy == "probs":
                priority = gate
            elif self.moe_token_drop_policy == "position":
                priority = (mask >= 0) * torch.arange(S, 0, -1, device=mask.device).view(hidden_states.shape[1], hidden_states.shape[0], 1).transpose(0, 1)
            else:
                raise NotImplementedError(f"unsupported MoE token drop policy \"{self.moe_token_drop_policy}\"")
            priority = priority.view(S, priority.shape[-1])  # bsE->SE
            capacity_indices = priority.topk(expert_capacity, dim=0).indices
            capacity_mask = torch.zeros_like(priority).scatter(0, capacity_indices, 1)
            capacity_mask = capacity_mask.view(*hidden_states.shape[:2], capacity_mask.shape[-1])
            gate = gate * capacity_mask
        experts_output = []
        for expert in self.experts:
            experts_output.append(expert(hidden_states))
        experts_output = torch.stack(experts_output, dim=-2)
        hidden_states = torch.einsum("bsE,bsEh->bsh", gate, experts_output)
        if self.shared_expert_hidden_size:
            if self.shared_expert_combine_method == "add":
                hidden_states = hidden_states + self.shared_expert(identity)
            elif self.shared_expert_combine_method == "softmax":
                coef = self.coefficient(identity).softmax(dim=-1)
                hidden_states = hidden_states * coef[..., :1] + self.shared_expert(identity) * coef[..., 1:]
            else:
                raise NotImplementedError(f"not implemented {self.shared_expert_combine_method=}")
        return hidden_states


def _llama_mlp_switch(config):
    global _LAYER_NUMBER
    _LAYER_NUMBER += 1
    moe_first = getattr(config, "moe_first", False)
    if getattr(config, "num_experts", None) is not None and (_LAYER_NUMBER - moe_first) % config.moe_layer_interval == 0:
        return LlamaMoE(config)
    else:
        return _LlamaMLP(config)


@contextlib.contextmanager
def replace_mlp_with_moe(mod):
    global _LlamaMLP
    global _LAYER_NUMBER
    _LlamaMLP = mod.LlamaMLP
    _LAYER_NUMBER = 0
    mod.LlamaMLP = _llama_mlp_switch
    yield
    mod.LlamaMLP = _LlamaMLP


def set_aux_loss_store(new_store):
    global _AUX_LOSS_STORE
    _AUX_LOSS_STORE = new_store


def get_aux_loss_store():
    return _AUX_LOSS_STORE
