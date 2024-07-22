
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)

def _wrap_embed_layer(layer: torch.nn.Module):
    layer.__class__ = EmbeddingPipe
    return layer


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx, activation_checkpointing=False):
        super().__init__(config, layer_idx)
        self.activation_checkpointing = activation_checkpointing

    # def __init__(self, config: LlamaConfig, activation_checkpointing=False):
    #     super().__init__(config)
    #     self.activation_checkpointing = activation_checkpointing

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            # attention_mask,
                                            None,
                                            position_ids,
        )
        return (outputs[0], position_ids, mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)
            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
        )

        return (outputs, position_ids, mask)


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        # print("#############################################")
        # print(type(args))
        # print(args.shape)
        # hidden_states = args
        if isinstance(args, torch.Tensor):
            hidden_states = args
        else:
            hidden_states, = args
        # print(hidden_states.shape)
        logits = super().forward(hidden_states)
        return (logits,)


def loss_fn(outputs, labels):
    # unpack
    logits, = outputs
    # all labels are `ignore_index` will cause nan
    if labels is None:
        return torch.tensor(0.1).to(logits.device)
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )

def loss_fn_reward(outputs, data_tmp):
    # unpack
    logits, = outputs
    rewards = logits.squeeze(-1)
    input_ids = data_tmp[0]
    PAD_ID, num_padding_at_beginning = int(data_tmp[1][0]), int(data_tmp[1][1]) 
    # print(f"loss_fn_reward input_ids:{input_ids}, type:{input_ids.shape}")
    # print(f"loss_fn_reward score:{rewards}, type:{rewards.shape}")
    chosen_mean_scores = []
    rejected_mean_scores = []

    # Split the inputs and rewards into two parts, chosen and rejected
    assert len(input_ids.shape) == 2
    bs = input_ids.shape[0] // 2
    seq_len = input_ids.shape[1]

    chosen_ids = input_ids[:bs]  #  1 * len
    rejected_ids = input_ids[bs:]  
    chosen_rewards = rewards[:bs]  # 1 * len
    rejected_rewards = rewards[bs:]

    # Compute pairwise loss. Only backprop on the different tokens before padding
    loss = 0
    for i in range(bs):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        chosen_reward = chosen_rewards[i]
        rejected_reward = rejected_rewards[i]

        c_inds = (chosen_id == PAD_ID).nonzero()
        c_ind = c_inds[num_padding_at_beginning].item() if len(
                c_inds
            ) > num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
        check_divergence = (chosen_id != rejected_id).nonzero()

        if len(check_divergence) == 0:
            end_ind = rejected_reward.size(-1)
            divergence_ind = end_ind - 1
            r_ind = c_ind
        else:
            # Check if there is any padding otherwise take length of sequence
            r_inds = (rejected_id == PAD_ID).nonzero()
            r_ind = r_inds[num_padding_at_beginning].item(
                ) if len(r_inds) > num_padding_at_beginning else seq_len
            end_ind = max(c_ind, r_ind)
            divergence_ind = check_divergence[0]

        assert divergence_ind > 0
        c_truncated_reward = chosen_reward[divergence_ind:end_ind]
        r_truncated_reward = rejected_reward[divergence_ind:end_ind]
        chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
        rejected_mean_scores.append(rejected_reward[r_ind - 1])
        c_truncated_reward = c_truncated_reward.float()
        r_truncated_reward = r_truncated_reward.float()
        loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
    loss = loss / bs
    chosen_mean_scores = torch.stack(chosen_mean_scores)
    rejected_mean_scores = torch.stack(rejected_mean_scores)
    return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }
        # return loss

def loss_fn_reward_value(outputs, data_tmp):
    # unpack
    logits, = outputs
    values = logits.squeeze(-1)
    input_ids = data_tmp[0]
    PAD_ID, prompt_length = int(data_tmp[1][0]), int(data_tmp[1][1]) 

    assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
    bs = values.size(0)
    seq_len = input_ids.shape[1]
    chosen_end_scores = []  # we use this name for consistency with the original forward function
    for i in range(bs):
        input_id = input_ids[i]
        value = values[i]
        c_inds = (input_id[prompt_length:] == PAD_ID).nonzero()
        # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
        c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
        chosen_end_scores.append(value[c_ind - 1])
    return {    
                "loss": torch.Tensor(1).to(logits.device),
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

    input_ids = data_tmp[0]
    PAD_ID, num_padding_at_beginning = int(data_tmp[1][0]), int(data_tmp[1][1]) 
    # print(f"loss_fn_reward input_ids:{input_ids}, type:{input_ids.shape}")
    # print(f"loss_fn_reward score:{rewards}, type:{rewards.shape}")
    chosen_mean_scores = []
    rejected_mean_scores = []

    # Split the inputs and rewards into two parts, chosen and rejected
    assert len(input_ids.shape) == 2
    bs = input_ids.shape[0] // 2
    seq_len = input_ids.shape[1]

    chosen_ids = input_ids[:bs]  #  1 * len
    rejected_ids = input_ids[bs:]  
    chosen_rewards = rewards[:bs]  # 1 * len
    rejected_rewards = rewards[bs:]

    # Compute pairwise loss. Only backprop on the different tokens before padding
    loss = 0
    for i in range(bs):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        chosen_reward = chosen_rewards[i]
        rejected_reward = rejected_rewards[i]

        c_inds = (chosen_id == PAD_ID).nonzero()
        c_ind = c_inds[num_padding_at_beginning].item() if len(
                c_inds
            ) > num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
        check_divergence = (chosen_id != rejected_id).nonzero()

        if len(check_divergence) == 0:
            end_ind = rejected_reward.size(-1)
            divergence_ind = end_ind - 1
            r_ind = c_ind
        else:
            # Check if there is any padding otherwise take length of sequence
            r_inds = (rejected_id == PAD_ID).nonzero()
            r_ind = r_inds[num_padding_at_beginning].item(
                ) if len(r_inds) > num_padding_at_beginning else seq_len
            end_ind = max(c_ind, r_ind)
            divergence_ind = check_divergence[0]

        assert divergence_ind > 0
        c_truncated_reward = chosen_reward[divergence_ind:end_ind]
        r_truncated_reward = rejected_reward[divergence_ind:end_ind]
        chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
        rejected_mean_scores.append(rejected_reward[r_ind - 1])
        c_truncated_reward = c_truncated_reward.float()
        r_truncated_reward = r_truncated_reward.float()
        loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
    loss = loss / bs
    chosen_mean_scores = torch.stack(chosen_mean_scores)
    rejected_mean_scores = torch.stack(rejected_mean_scores)
    return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }


def get_model(model_config: LlamaConfig, args, activation_checkpointing_config=None, **kwargs):
    class GPT2ModelPipe(PipelineModule):
        def __init__(self, model_config, **kwargs):
            if activation_checkpointing_config:
                deepspeed.checkpointing.configure(
                    None,
                    partition_activations=activation_checkpointing_config.get("partition_activations", False),
                    contiguous_checkpointing=activation_checkpointing_config.get("contiguous_memory_optimization", False),
                    checkpoint_in_cpu=activation_checkpointing_config.get("cpu_checkpointing", False),
                    num_checkpoints=activation_checkpointing_config.get("number_checkpoints", None),
                    synchronize=activation_checkpointing_config.get("synchronize_checkpoint_boundary", False),
                    profile=activation_checkpointing_config.get("profile", False),
                )
            super().__init__(
                layers=[
                    LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
                    *[LayerSpec(ParallelTransformerLayerPipe, model_config, _)
                        for _ in range(model_config.num_hidden_layers)],
                    LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
                    LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
                ],
                activation_checkpoint_interval=(1 if activation_checkpointing_config else 0),
                checkpointable_layers=["ParallelTransformerLayerPipe"],
                **kwargs
            )

    pp = args.pipe_parallel_size
    mp = args.model_parallel_size
    assert args.world_size % (pp * mp) == 0
    dp = args.world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
    # Offset base seeds for the interior pipeline stages.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim('pipe') - 1:
        args.seed = args.seed + (stage_id * mp)

    return GPT2ModelPipe(model_config,
                         loss_fn=loss_fn,
                         topology=topo,
                         base_seed=args.seed,
                         **kwargs)

def get_reward_model(model_config: LlamaConfig, args, activation_checkpointing_config=None, **kwargs):
    class GPT2ModelPipe(PipelineModule):
        def __init__(self, model_config, **kwargs):
            if activation_checkpointing_config:
                deepspeed.checkpointing.configure(
                    None,
                    partition_activations=activation_checkpointing_config.get("partition_activations", False),
                    contiguous_checkpointing=activation_checkpointing_config.get("contiguous_memory_optimization", False),
                    checkpoint_in_cpu=activation_checkpointing_config.get("cpu_checkpointing", False),
                    num_checkpoints=activation_checkpointing_config.get("number_checkpoints", None),
                    synchronize=activation_checkpointing_config.get("synchronize_checkpoint_boundary", False),
                    profile=activation_checkpointing_config.get("profile", False),
                )
            super().__init__(
                layers=[
                    LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
                    *[LayerSpec(ParallelTransformerLayerPipe, model_config, _)
                        for _ in range(model_config.num_hidden_layers)],
                    LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
                    LayerSpec(LMLayerPipe, model_config.hidden_size, 1, bias=False),
                ],
                activation_checkpoint_interval=(1 if activation_checkpointing_config else 0),
                checkpointable_layers=["ParallelTransformerLayerPipe"],
                **kwargs
            )

    pp = args.pipe_parallel_size
    mp = args.model_parallel_size
    assert args.world_size % (pp * mp) == 0
    dp = args.world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
    # Offset base seeds for the interior pipeline stages.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim('pipe') - 1:
        args.seed = args.seed + (stage_id * mp)
    # barrier
    return GPT2ModelPipe(model_config,
                         loss_fn=loss_fn_reward,
                         topology=topo,
                         base_seed=args.seed,
                         **kwargs)

