from typing import List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.nn.functional as F
# from .modeling_yi_sliding_windows import YiDecoderLayer, YiRMSNorm, YiConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2Config, logger, Qwen2MLP, \
     QWEN2_ATTENTION_CLASSES, Qwen2FlashAttention2,pad_input,flash_attn_varlen_func
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, ProcessTopology
import math

######################################sequence parallel set up#################################################
_SEQUENCE_PARALLEL_GROUP = None

def initialize_model_parallel(args):
    # parallel check
    pp = args.pipe_parallel_size
    mp = args.model_parallel_size
    sp = args.sequence_parllel_size
    assert args.world_size % (pp * mp * sp) == 0
    dp = args.world_size // (pp * mp * sp)
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    print(f"world_size:{args.world_size}")
    num_sequence_parallel_groups  = args.world_size // pp // mp // sp * pp  # 待验证
    rank = torch.distributed.get_rank()
    print("num_sequence_parallel_groups:{}".format(num_sequence_parallel_groups))
    # logger.info(f" torch.distributed.get_rank:{rank}")
    # notes：集群设置要注意
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp, (i + 1) * sp)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            global _SEQUENCE_PARALLEL_GROUP
            _SEQUENCE_PARALLEL_GROUP = group   
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_world_size():
    """Get the sequence parallel world size."""
    return dist.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank():
    """Get the sequence parallel rank."""
    return dist.get_rank(group=get_sequence_parallel_group())

######################################sequence parallel#################################################


class PipeModelDataSequenceParallelTopology(ProcessTopology):
    """ A topology for hybrid pipeline, model, and data parallelism. """

    def __init__(self, num_pp, num_mp, num_dp, num_sp=1):
        super().__init__(axes=['pipe', 'data', 'model', 'sequence'], dims=[num_pp, num_dp, num_mp, num_sp])

class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)

def _wrap_embed_layer(layer: torch.nn.Module):
    layer.__class__ = EmbeddingPipe
    return layer


class Qwen2FlashAttention2Sp(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func_dist(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale,
                    causal,
                )
            else:
                attn_output = flash_attn_func_dist(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale,
                    causal,
                    (self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output



QWEN2_ATTENTION_CLASSES["flash_attention_2_sp"] = Qwen2FlashAttention2Sp

class Qwen2DecoderLayerSp(torch.nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.self_attn = Qwen2FlashAttention2Sp(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # if "padding_mask" in kwargs:
        #     warnings.warn(
        #         "Passing `padding_mask` is deprecated and will be removed in v4.37. "
        #         "Please make sure use `attention_mask` instead.`"
        #     )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class ParallelTransformerLayerPipe(Qwen2DecoderLayerSp):
    def __init__(self, config: Qwen2Config, idx, activation_checkpointing=False):
        super().__init__(config, idx)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, position_ids, mask = args
        # attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        outputs = Qwen2DecoderLayerSp.forward(self,
                                            hidden_states,
                                            None,
                                            position_ids,
        )
        return (outputs[0], position_ids, mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return Qwen2DecoderLayerSp.forward(module, *inputs)
            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
        )

        return (outputs, position_ids, mask)

class LayerNormPipe(Qwen2RMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states = args
        logits = super().forward(hidden_states)
        return (logits,)


def loss_fn(outputs, labels):
    # unpack
    logits, = outputs
    # all labels are `ignore_index` will cause nan
    tmp = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1)
    )
    if math.isnan(tmp):
        a = 3
    return tmp

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

def get_model(model_config: Qwen2Config, args, activation_checkpointing_config=None, **kwargs):
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
    sp = args.sequence_parllel_size
    assert args.world_size % (pp * mp * sp) == 0
    dp = args.world_size // (pp * mp * sp)

    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
    # topo = PipeModelDataSequenceParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp, num_sp=sp)
    # Offset base seeds for the interior pipeline stages.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim('pipe') - 1:
        args.seed = args.seed + (stage_id * mp)
    
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    if args.use_sp:
        try:
            from deepspeed.sequence.layer import DistributedAttention

            global flash_attn_func_dist
            flash_attn_func_dist = DistributedAttention(flash_attn_func, _SEQUENCE_PARALLEL_GROUP)
        except:
            logger.error("use DistributedAttention failed...")

    return GPT2ModelPipe(model_config,
                         loss_fn=loss_fn,
                         topology=topo,
                         base_seed=args.seed,
                         **kwargs)

