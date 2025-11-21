import math
import torch
import torch.nn.functional as F

from transformers.models.llama import configuration_llama, modeling_llama

try:
    from . import moe_reference
except ImportError:  # if __name__ == "__main__":
    import moe_reference


class CosineLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, lr, min_lr, decay_iters, warmup_iters):
        self.lr = lr
        self.min_lr = min_lr
        self.decay_iters = decay_iters
        self.warmup_iters = warmup_iters
        super().__init__(optimizer)

    def get_lr(self):
        res = [self.lr * self.last_epoch / self.warmup_iters if self.last_epoch <= self.warmup_iters
                else self.min_lr if self.last_epoch >= self.decay_iters
                else self.min_lr + (self.lr - self.min_lr) / 2 * (1 + math.cos((self.last_epoch - self.warmup_iters) / (self.decay_iters - self.warmup_iters) * math.pi))
                for _ in self.base_lrs]
        return res


class Llama:
    def init_model(self, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, max_position_embeddings, rms_norm_eps, rope_theta,
                   num_experts, expert_hidden_size, shared_expert_hidden_size, shared_expert_combine_method, moe_layer_interval, moe_first, moe_router_topk, moe_aux_loss_coeff,
                   moe_expert_capacity_factor, moe_token_drop_policy, torch_dtype):
        P = num_hidden_layers * (2 * hidden_size + 2 * num_key_value_heads * (hidden_size // num_attention_heads) + 3 * intermediate_size + 2) * hidden_size + hidden_size + 2 * vocab_size * hidden_size
        if 20 * P > 80 * 1024 * 1024 * 1024:
            raise RuntimeError(f"Number of parameters {P=} is too large")
        config = configuration_llama.LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            # hidden_act="silu",
            max_position_embeddings=max_position_embeddings,
            # initializer_range=0.02,
            rms_norm_eps=rms_norm_eps,
            # use_cache=True,
            # pad_token_id=None,
            # bos_token_id=1,
            # eos_token_id=2,
            # pretraining_tp=1,
            # tie_word_embeddings=False,
            rope_theta=rope_theta,
            # rope_scaling=None,
            # attention_bias=False,
            # attention_dropout=0.0,
            # **kwargs,
            num_experts=num_experts,
            expert_hidden_size=expert_hidden_size,
            shared_expert_hidden_size=shared_expert_hidden_size,
            shared_expert_combine_method=shared_expert_combine_method,
            moe_layer_interval=moe_layer_interval,
            moe_first=moe_first,
            moe_router_topk=moe_router_topk,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_token_drop_policy=moe_token_drop_policy,
            torch_dtype=torch_dtype,
        )
        self.config = config
        with moe_reference.replace_mlp_with_moe(modeling_llama):
            self.model = modeling_llama.LlamaForCausalLM(config)
        self.model = self.model.cuda()
        self.amp_dtype = torch_dtype

    def init_optimizer(self, lr, min_lr, lr_decay_iters, lr_warmup_iters, weight_decay, optimizer, optimizer_kwargs, num_microbatches, max_norm):
        param_groups = [
            {
                "weight_decay": weight_decay,
                "params": [param for name, param in self.model.named_parameters() if len(param.shape) != 1],
            },
            {
                "weight_decay": 0.,
                "params": [param for name, param in self.model.named_parameters() if len(param.shape) == 1],  # do not regularize Norm parameters
            },
        ]
        if optimizer == "adam":
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, **optimizer_kwargs, fused=True)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, lr=lr, **optimizer_kwargs)
        else:
            raise NotImplementedError(f"not implemented optimizer \"{optimizer}\"")
        self.lr_scheduler = CosineLRScheduler(self.optimizer, lr, min_lr, lr_decay_iters, lr_warmup_iters)
        self.iteration = 0
        self.num_microbatches = num_microbatches
        self.micro_batch_cnt = 0
        self.max_norm = max_norm
        self.sum_lm_loss = torch.tensor(0, dtype=torch.float, device="cuda")
        self.sum_aux_loss = torch.tensor(0, dtype=torch.float, device="cuda")

    def init_tensorboard(self, writer, tensorboard_log_interval):
        self.writer = writer
        self.tensorboard_log_interval = tensorboard_log_interval

    def feed_input_data(self, input_ids, labels):
        if self.config.num_experts is not None:
            moe_reference.set_aux_loss_store([])
        with torch.autocast("cuda", dtype=self.amp_dtype):
            logits = self.model(input_ids, return_dict=True).logits
        lm_loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction='none')
        lm_loss = lm_loss.sum() / labels.numel()
        self.sum_lm_loss += lm_loss.detach() / self.num_microbatches
        if self.config.num_experts is not None:
            aux_loss_store = moe_reference.get_aux_loss_store()
            assert len(aux_loss_store) == self.config.num_hidden_layers // self.config.moe_layer_interval + \
                (self.config.moe_first and self.config.num_hidden_layers % self.config.moe_layer_interval != 0)
            aux_loss = torch.sum(torch.stack(aux_loss_store))
            moe_reference.set_aux_loss_store(None)
            self.sum_aux_loss += aux_loss.detach() / len(aux_loss_store) / self.num_microbatches
            loss = lm_loss + self.config.moe_aux_loss_coeff * aux_loss
        else:
            loss = lm_loss
        loss = loss / self.num_microbatches
        loss.backward()
        self.micro_batch_cnt += 1

    def step(self):
        assert self.micro_batch_cnt == self.num_microbatches, "number of micro-batches mismatch"
        self.iteration += 1
        lm_loss = self.sum_lm_loss.item()
        aux_loss = self.sum_aux_loss.item()
        grad_norm = sum([(param.grad ** 2).sum().item() for param in self.model.parameters()]) ** .5
        s = "golden model:"
        s += f" lm loss {lm_loss}"
        if self.config.num_experts is not None:
            s += f" aux_loss {aux_loss}"
        s += f" grad_norm {grad_norm}"
        print(s)
        if self.writer is not None and self.iteration % self.tensorboard_log_interval == 0:
            self.writer.add_scalars("lm loss", {"reference": lm_loss}, self.iteration)
            if self.config.num_experts is not None:
                self.writer.add_scalars("load_balancing_loss", {"reference": aux_loss}, self.iteration)
            self.writer.add_scalars("grad-norm", {"reference": grad_norm}, self.iteration)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.micro_batch_cnt = 0
        self.sum_lm_loss.zero_()
        self.sum_aux_loss.zero_()


if __name__ == "__main__":
    vocab_size = 128000
    seq_length = 2048
    llama = Llama()
    llama.init_model(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=2688,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=seq_length,
        rms_norm_eps=1e-5,
        rope_theta=10000.,
        num_experts=4,
        expert_hidden_size=640,
        shared_expert_hidden_size=1280,
        shared_expert_combine_method="softmax",
        moe_layer_interval=2,
        moe_first=False,
        moe_router_topk=2,
        moe_aux_loss_coeff=1e-2,
        moe_expert_capacity_factor=1.,
        moe_token_drop_policy="probs",
        torch_dtype=torch.bfloat16,
    )
    num_microbatches = 4
    llama.init_optimizer(3e-4, 3e-5, 10000, 50, .1, "sgd", {"momentum": .9}, num_microbatches, 1.)
    llama.init_tensorboard(None, 1)
    for iteration in range(10):
        for _ in range(num_microbatches):
            text = (torch.rand(1, seq_length + 1) * vocab_size).to(torch.long).cuda()
            input_ids = text[:, :-1]
            labels = text[:, 1:]
            llama.feed_input_data(input_ids, labels)
        llama.step()
