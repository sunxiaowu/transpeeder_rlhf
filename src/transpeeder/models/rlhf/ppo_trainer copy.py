# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import copy
import time
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from .utils import print_rank_0, get_position_ids, get_attn_mask
from ..qwen1_5_pipeline_model_sp import loss_fn_reward_value, loss_fn_reward


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))   
    return log_probs_labels.squeeze(-1)

def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = -100
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def actor_loss_fn_auxiliary(logprobs, old_logprobs, advantages, mask, cliprange):
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

def critic_loss_fn_auxiliary(values, old_values, returns, mask, cliprange_value):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - cliprange_value,
            old_values + cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

def actor_loss_fn(outputs, args):
    start = int(args[1])
    seq = args[0]
    log_probs = args[2]
    advantages = args[3]
    action_mask = args[4]
    cliprange = float(args[5])
    
    actor_prob, = outputs
    actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
    actor_loss = actor_loss_fn_auxiliary(actor_log_prob[:, start:],
                                            log_probs[:, start:], advantages,
                                            action_mask[:, start:], cliprange)
    return actor_loss
    
def critic_loss_fn(outputs, args):
    #    (
    #              input_ids,
    #              torch.tensor(start),
    #              old_values,
    #              returns,
    #              action_mask,
    #              torch.tensor(cliprange_value),
    #              torch.tensor(pad_token_id),
    #              torch.tensor(prompt_len)
    #         )
    seq = args[0]
    start = int(args[1]) + 1
    old_values = args[2]
    returns = args[3]
    action_mask = args[4]
    cliprange_value = float(args[5])
    pad_token_id = args[6]
    prompt_len = args[7]
    data_tmp = [seq, [pad_token_id, prompt_len]]
    values = loss_fn_reward_value(outputs, data_tmp)["values"]
    critic_loss = critic_loss_fn_auxiliary(values[:, start:],values[:, start:], returns, action_mask[:, start:], cliprange_value)
 # critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
        #                                                                start:],
        #                                   returns, action_mask[:, start:])
    return critic_loss

def dpo_loss_fn(outputs, args):
    a = 3
    labels = args[0]
    reference_chosen_logps = args[1]
    reference_rejected_logps = args[2]

    

class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer_actor
        self.pad_token_id = self.tokenizer.pad_token_id
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step, tokenizer):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        kwargs = dict(do_sample=False)
        # with torch.no_grad():
        #     seq = self.actor_model.module.generate(
        #         prompts,
        #         attention_mask=mask,
        #         max_length=max_min_length,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         synced_gpus=self.z3_enabled,
        #         **kwargs)
        seq, out_logprobs = self.actor_model.generate(
                prompt_tokens=prompts.tolist(),
                # attention_mask=mask,
                max_gen_len=self.args.max_answer_seq_len,
                temperature=0.6,
                top_p=0.9,
                logprobs=False,
                echo=True,
                tokenizer=tokenizer)
        

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        # ans = seq
        valid_ans_len = (ans != tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers:
            print_rank_0(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print_rank_0(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(ans, skip_special_tokens=True)}"
             )
            # print(
            #     f"--- full    --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(seq, skip_special_tokens=True)}"
            # )
            

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def _gen_data_iter_ppo_actor_loss(self, input_ids, start, log_probs, advantages, action_mask, cliprange):
        # start = args[0]
        # seq = args[1]
        # log_probs = args[2]
        # advantages = args[3]
        # action_mask = args[4]
        a = 3

        # tmp_list_tensor = torch.tensor([pad_token_id, prompt_len])
        # tmp_list_tensor = torch.tensor([])
        # start_tensor = torch.tensor(start)
        iter_data = [(
            (      
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 torch.tensor(start),
                 log_probs,
                 advantages,
                 action_mask,
                 torch.tensor(cliprange)
            )
            )]
        generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        return generate_iter
    
    def _gen_data_iter_ppo_critic_loss(self, input_ids, start, old_values, returns, action_mask, cliprange_value, pad_token_id, prompt_len):
        iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 torch.tensor(start),
                 old_values,
                 returns,
                 action_mask,
                 torch.tensor(cliprange_value),
                 torch.tensor(pad_token_id),
                 torch.tensor(prompt_len)
            )
            )]
        generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        return generate_iter

         
    def _gen_data_iter_train(self, mode="train", input_ids=None, pad_token_id=None, prompt_len=512, num_padding_at_beginning=1):
        if mode == 'train':
            iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids
            )
            )]
            generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        elif mode == "eval":
            generate_iter = None
        elif mode == "generate":
            generate_iter = None
        elif mode == "eval_reward":
            tmp_list_tensor = torch.tensor([pad_token_id, prompt_len])
            iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 tmp_list_tensor
            )
            )]
            generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        else:
            generate_iter = None
        return generate_iter


    def generate_experience(self, prompts, mask, prompts_critic, mask_critic, step, tokenizer, tokenizer_critic):
        self.eval_pp()
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step, tokenizer)  # [batch_size, prompt_response_max_len]
        generate_end = time.time()
        self.train_pp()
        pad_token_id = tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():

            logits_buffer_actor_buffer = torch.Tensor(seq.shape[0], seq.shape[1], self.args.model_actor_config.vocab_size).to(self.args.local_rank)
            logits_buffer_ref_buffer = torch.Tensor(seq.shape[0], seq.shape[1], self.args.model_actor_config.vocab_size).to(self.args.local_rank)
            reward_buffer = torch.Tensor(seq.shape[0]).to(self.args.local_rank)
            values_buffer = torch.Tensor(seq.shape[0], seq.shape[1]).to(self.args.local_rank)

            logits = torch.zeros_like(logits_buffer_actor_buffer)
            rel_logits = torch.zeros_like(logits_buffer_ref_buffer)
            reward_score = torch.zeros_like(reward_buffer)
            values = torch.zeros_like(values_buffer)

            data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
            actor_loss, outputs = self.actor_model.eval_batch(data_iter=data_iter, return_logits=True, mode="train")
            if self.actor_model.is_last_stage():
                logits, = outputs
                logits = logits.to(torch.float32)
           
            rel_loss, rel_outputs = self.ref_model.eval_batch(data_iter=data_iter, return_logits=True, mode="eval")
            if self.ref_model.is_last_stage():
                rel_logits, = rel_outputs
                rel_logits = rel_logits.to(torch.float32)

            # dist.barrier()
            dist.all_reduce(logits, group=self.actor_model.mpu.get_pipe_parallel_group())
            # logits = copy.deepcopy(logits)
            # print("{} -> {}\n".format(self.args.local_rank, logits))
            # print("{} -> {}\n".format(self.args.local_rank, rel_logits))
            # logprobs = copy.deepcopy(gather_log_probs(logits[:, :-1, :], seq[:, 1:]).detach())
            # dist.barrier()
            a = 3
            dist.all_reduce(rel_logits, group=self.ref_model.mpu.get_pipe_parallel_group())
            # print("{} -> {}\n".format(self.args.local_rank, logits))
            # print("{} -> {}\n".format(self.args.local_rank, rel_logits))
            # ref_logprobs =  gather_log_probs(rel_logits[:, :-1, :], seq[:, 1:])
            # dist.barrier()

            # 采用deepspeed pipeline方式加载
            if self.args.critic_use_pp:

                self.reward_model.module.loss_fn = loss_fn_reward_value
                data_iter = self._gen_data_iter_train(mode="eval_reward", input_ids=seq, pad_token_id=pad_token_id, prompt_len=prompts.shape[1])
                reward_loss, reward_outputs = self.reward_model.eval_batch(data_iter=data_iter, return_logits=True, mode="eval")

                if self.reward_model.is_last_stage():
                    # 注意此处，数据的分布式采集时最外层做的，此处传递给eval_batch的迭代只包含一个batch_per_gpu，迭代循环自己，deepspeed的梯度累积计算，结果是一样的，所以取结果一次就行
                    reward_score = self.reward_model.fwd_outputs_dict[0]["chosen_end_scores"].clone().detach().to(torch.float32)
                dist.all_reduce(reward_score, group=self.reward_model.mpu.get_pipe_parallel_group())

                self.critic_model.module.loss_fn = loss_fn_reward_value
                data_iter = self._gen_data_iter_train(mode="eval_reward", input_ids=seq, pad_token_id=pad_token_id, prompt_len=prompts.shape[1])
                critic_loss, critic_outputs = self.critic_model.eval_batch(data_iter=data_iter, return_logits=True,mode="train")
                if self.critic_model.is_last_stage():
                    # 同上
                    values = self.critic_model.fwd_outputs_dict[0]["values"].clone().detach().to(torch.float32)
                dist.all_reduce(values, group=self.critic_model.mpu.get_pipe_parallel_group())
                # 变量同步
                # dist.barrier()
                # dist.all_reduce(logits, group=self.actor_model.mpu.get_pipe_parallel_group())
                # dist.all_reduce(rel_logits, group=self.ref_model.mpu.get_pipe_parallel_group())
                # dist.all_reduce(reward_score, group=self.reward_model.mpu.get_pipe_parallel_group())
                # dist.all_reduce(values, group=self.critic_model.mpu.get_pipe_parallel_group())
                # a = 3
            
                # 变量同步
                # logits_list = [logits, rel_logits]
                # dist.barrier()
                # dist.all_reduce(logits_list, group=self.mpu.get_pipe_parallel_group())

                # self.
                #  output = self.actor_model(seq, attention_mask=attention_mask)    # [batch_size, len, vocab_size]
                # output_ref = self.ref_model(seq, attention_mask=attention_mask)  # [batch_size, len, vocab_size]
                # reward_score = self.reward_model.forward_value(
                #     seq, attention_mask,
                #     prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                #     )
                # values = self.critic_model.forward_value(
                #     seq, attention_mask, return_value_only=True).detach()[:, :-1]

                # logits = output.logits                   
                # rel_logits = output_ref.logits

            # 直接采用deepspeed stage方式加载
            else:
                # seq_encode = tokenizer_critic.batch_encode_plus(tokenizer.batch_decode(seq), return_tensors="pt")
                # reward_score = self.reward_model.forward_value(
                # seq_encode["input_ids"].cuda(), seq_encode["attention_mask"].cuda(),
                # prompt_length=self.prompt_length)['chosen_end_scores'].detach()
                seq_encode = seq    # 默认actor model 和 critic model采用相同tokenizer， 否则 用上面注释代码做转换， 目前qwen0.5（rm) 和 qwen 4b(actor)是一致的
                
                reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach()
                values = self.critic_model.forward_value(seq, attention_mask, return_value_only=True).detach()[:, :-1] 

        self.generate_time = generate_end - generate_start
    #        return {
    #     'prompts': prompts, # [batch_size, prompt_max_len]
    #     'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]), # [batch_size, prompt_response_max_len]
    #     'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]), # [batch_size, prompt_response_max_len]
    #     'value': values, # [batch_size, prompt_response_max_len]
    #     'rewards': reward_score, # [batch_size]
    #     'input_ids': seq, # [batch_size, prompt_response_max_len]
    #     "attention_mask": attention_mask # [batch_size, prompt_response_max_len]
    # }
        # dist.barrier()
        tmp = {
            'prompts': prompts,   # tensor (b,prompt_seq_len)
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),    # tensor (b,all_seq_len -1 )
            'ref_logprobs': gather_log_probs(rel_logits[:, :-1, :], seq[:,   # tensor (b,all_seq_len -1 )
                                                                        1:]),
            'value': values,         # tensor (b,all_seq_len -1 )
            'rewards': reward_score,   
            'input_ids': seq,              
            "attention_mask": attention_mask 
        }
        return {
            'prompts': prompts,   # tensor (b,prompt_seq_len)
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),    # tensor (b,all_seq_len -1 )
            'ref_logprobs': gather_log_probs(rel_logits[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,           
            'rewards': reward_score,     
            'input_ids': seq,                
            "attention_mask": attention_mask 
        }

    def generate_experienceBAK(self, prompts, mask, step, tokenizer):
        self.eval_pp()
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step, tokenizer)  # [batch_size, prompt_response_max_len]
        generate_end = time.time()
        self.train_pp()
        pad_token_id = tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            logits_buffer = torch.Tensor(seq.shape[0], seq.shape[1], tokenizer.vocab_size + 1).to(self.args.local_rank)
            reward_buffer = torch.Tensor(seq.shape[0], 1).to(self.args.local_rank)
            values_buffer = torch.Tensor(seq.shape[0], seq.shape[1]).to(self.args.local_rank)

            logits = torch.zeros_like(logits_buffer)
            rel_logits = torch.zeros_like(logits_buffer)
            reward_score = torch.zeros_like(reward_buffer)
            values = torch.zeros_like(values_buffer)

            data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
            actor_loss, outputs = self.actor_model.eval_batch(data_iter=data_iter, return_logits=True)
            # if self.actor_model.is_last_stage():
            #     logits, = outputs
           
            rel_loss, rel_outputs = self.ref_model.eval_batch(data_iter=data_iter, return_logits=True)
            # if self.ref_model.is_last_stage():
            #     rel_logits, = rel_outputs
   

            self.reward_model.module.loss_fn = loss_fn_reward_value
            data_iter = self._gen_data_iter_train(mode="eval_reward", input_ids=seq, pad_token_id=pad_token_id, prompt_len=prompts.shape[1])
            reward_loss, reward_outputs = self.reward_model.eval_batch(data_iter=data_iter, return_logits=True)

            # if self.reward_model.is_last_stage():
            #     # 注意此处，数据的分布式采集时最外层做的，此处传递给eval_batch的迭代只包含一个batch_per_gpu，迭代循环自己，deepspeed的梯度累积计算，结果是一样的，所以取结果一次就行
            #     reward_score = self.reward_model.fwd_outputs_dict[0]["chosen_end_scores"].view(1, -1).detach()


            self.critic_model.module.loss_fn = loss_fn_reward_value
            data_iter = self._gen_data_iter_train(mode="eval_reward", input_ids=seq, pad_token_id=pad_token_id, prompt_len=prompts.shape[1])
            critic_loss, critic_outputs = self.critic_model.eval_batch(data_iter=data_iter, return_logits=True)
            # if self.critic_model.is_last_stage():
            #     # 同上
            #     values = self.critic_model.fwd_outputs_dict[0]["values"].detach()

            if self.actor_model.is_last_stage() and self.ref_model.is_last_stage() and self.reward_model.is_last_stage() and self.critic_model.is_last_stage():
                logits, = outputs
                rel_logits, = rel_outputs
                reward_score = self.reward_model.fwd_outputs_dict[0]["chosen_end_scores"].detach()
                values = self.critic_model.fwd_outputs_dict[0]["values"].detach()
            # 

            # 变量同步
            dist.barrier()
            dist.all_reduce(logits, group=self.actor_model.mpu.get_pipe_parallel_group())
            dist.all_reduce(rel_logits, group=self.ref_model.mpu.get_pipe_parallel_group())
            dist.barrier()
            dist.all_reduce(reward_score, group=self.reward_model.mpu.get_pipe_parallel_group())
            dist.all_reduce(values, group=self.critic_model.mpu.get_pipe_parallel_group())
            # dist.barrier()
            a = 3


            # 变量同步
            # logits_list = [logits, rel_logits]
            # dist.barrier()
            # dist.all_reduce(logits_list, group=self.mpu.get_pipe_parallel_group())

            # self.
           #  output = self.actor_model(seq, attention_mask=attention_mask)    # [batch_size, len, vocab_size]
            # output_ref = self.ref_model(seq, attention_mask=attention_mask)  # [batch_size, len, vocab_size]
            # reward_score = self.reward_model.forward_value(
            #     seq, attention_mask,
            #     prompt_length=self.prompt_length)['chosen_end_scores'].detach(
            #     )
            # values = self.critic_model.forward_value(
            #     seq, attention_mask, return_value_only=Tru5e).detach()[:, :-1]

        # logits = output.logits
        # rel_logits = output_ref.logits

        self.generate_time = generate_end - generate_start
    #        return {
    #     'prompts': prompts, # [batch_size, prompt_max_len]
    #     'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]), # [batch_size, prompt_response_max_len]
    #     'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]), # [batch_size, prompt_response_max_len]
    #     'value': values, # [batch_size, prompt_response_max_len]
    #     'rewards': reward_score, # [batch_size, 1]
    #     'input_ids': seq, # [batch_size, prompt_response_max_len]
    #     "attention_mask": attention_mask # [batch_size, prompt_response_max_len]
    # }
        tmp = {
            'prompts': prompts,   # tensor (b,prompt_seq_len)
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),    # tensor (b,all_seq_len -1 )
            'ref_logprobs': gather_log_probs(rel_logits[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,         
            'rewards': reward_score,   
            'input_ids': seq,              
            "attention_mask": attention_mask 
        }
        return {
            'prompts': prompts,   # tensor (b,prompt_seq_len)
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),    # tensor (b,all_seq_len -1 )
            'ref_logprobs': gather_log_probs(rel_logits[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,           
            'rewards': reward_score,     
            'input_ids': seq,                
            "attention_mask": attention_mask 
        }


    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            # old rewards   [batchsize, len(seq) - 1]
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
                # advantages, returns  (batchsize, values.shape[-1] - start)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        # batch = {'input_ids': seq, "attention_mask": attention_mask}
        # actor_prob = self.actor_model(**batch, use_cache=False).logits

        # data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
        data_iter_actor = self._gen_data_iter_ppo_actor_loss(input_ids=seq, start=start, log_probs=log_probs, advantages=advantages, action_mask=action_mask, cliprange=self.cliprange)
        # 切换loss计算函数，计算后反向传播
        self.actor_model.module.loss_fn = actor_loss_fn
        actor_loss = self.actor_model.train_batch(data_iter=data_iter_actor)

        a = 3
        # if self.actor_model.is_last_stage():
        #     actor_prob, = outputs
        #     actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        #     actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
        #                                     log_probs[:, start:], advantages,
        #                                     action_mask[:, start:])

            # if not self.args.align_overflow:
            #     self.actor_model.step()

        # value = self.critic_model.forward_value(**batch,
        #                                         return_value_only=True,
        #                                         use_cache=False)[:, :-1]


        # exit(0)
      
      

        if self.args.critic_use_pp:
            # to do
            data_iter_critic = self._gen_data_iter_ppo_critic_loss(input_ids=seq, start=start, old_values=old_values, returns=returns, 
                                                               action_mask=action_mask,
                                                                cliprange_value=self.cliprange_value, 
                                                                pad_token_id=self.pad_token_id,
                                                                prompt_len=prompts.shape[1])
            self.critic_model.module.loss_fn = critic_loss_fn
            critic_loss, outputs = self.critic_model.train_batch(data_iter=data_iter_critic)
            pass
        else:
            value = self.critic_model.forward_value(input_ids=seq,
                                                     attention_mask=attention_mask,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
            critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
            self.critic_model.backward(critic_loss)
            a = 4

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            # self.actor_model.step()

        a = 3
        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def train_pp(self):
        self.actor_model.module.train()
        self.critic_model.module.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def eval_pp(self):
        self.actor_model.module.eval()
        self.critic_model.module.eval()
        self.reward_model.module.eval()
        self.ref_model.module.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedDPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.tokenizer = self.rlhf_engine.tokenizer_actor
        self.pad_token_id = self.tokenizer.pad_token_id
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        # self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.reference_free = args.reference_free
        self.label_smoothing = args.label_smoothing
        self.ftx_gamma = args.ftx_gamma

        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step, tokenizer):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        kwargs = dict(do_sample=False)
        # with torch.no_grad():
        #     seq = self.actor_model.module.generate(
        #         prompts,
        #         attention_mask=mask,
        #         max_length=max_min_length,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         synced_gpus=self.z3_enabled,
        #         **kwargs)
        seq, out_logprobs = self.actor_model.generate(
                prompt_tokens=prompts.tolist(),
                # attention_mask=mask,
                max_gen_len=self.args.max_answer_seq_len,
                temperature=0.6,
                top_p=0.9,
                logprobs=False,
                echo=True,
                tokenizer=tokenizer)
        

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        # ans = seq
        valid_ans_len = (ans != tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers:
            print_rank_0(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print_rank_0(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(ans, skip_special_tokens=True)}"
             )
            # print(
            #     f"--- full    --> step={step}, rank={torch.distributed.get_rank()}, {tokenizer.batch_decode(seq, skip_special_tokens=True)}"
            # )
            

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def _gen_data_iter_dpo_actor_loss(self, input_ids, labels, reference_chosen_logps, reference_rejected_logps):
        iter_data = [
            (
            (      
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                labels,
                reference_chosen_logps,
                reference_rejected_logps
            )
            )
        ]
        generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        return generate_iter

    def _gen_data_iter_ppo_actor_loss(self, input_ids, start, log_probs, advantages, action_mask, cliprange):
        # start = args[0]
        # seq = args[1]
        # log_probs = args[2]
        # advantages = args[3]
        # action_mask = args[4]
        a = 3

        # tmp_list_tensor = torch.tensor([pad_token_id, prompt_len])
        # tmp_list_tensor = torch.tensor([])
        # start_tensor = torch.tensor(start)
        iter_data = [(
            (      
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 torch.tensor(start),
                 log_probs,
                 advantages,
                 action_mask,
                 torch.tensor(cliprange)
            )
            )]
        generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        return generate_iter
    
    def _gen_data_iter_ppo_critic_loss(self, input_ids, start, old_values, returns, action_mask, cliprange_value, pad_token_id, prompt_len):
        iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 torch.tensor(start),
                 old_values,
                 returns,
                 action_mask,
                 torch.tensor(cliprange_value),
                 torch.tensor(pad_token_id),
                 torch.tensor(prompt_len)
            )
            )]
        generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        return generate_iter

         
    def _gen_data_iter_train(self, mode="train", input_ids=None, pad_token_id=None, prompt_len=512, num_padding_at_beginning=1):
        if mode == 'train':
            iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids
            )
            )]
            generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        elif mode == "eval":
            generate_iter = None
        elif mode == "generate":
            generate_iter = None
        elif mode == "eval_reward":
            tmp_list_tensor = torch.tensor([pad_token_id, prompt_len])
            iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                 input_ids,
                 tmp_list_tensor
            )
            )]
            generate_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
        else:
            generate_iter = None
        return generate_iter

    def post_process(self, input_ids, all_logits, labels):
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=labels)
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = input_ids.size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.detach()
            rejected_rewards = self.beta * policy_rejected_logps.detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        # pi_logratios = pi_logratios
        # ref_logratios = ref_logratios
        pi_logratios = pi_logratios
        ref_logratios = ref_logratios
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps - reference_chosen_logps
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps
                - reference_rejected_logps
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards


    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    # def concatenated_forward(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    #     r"""
    #     Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

    #     Otherwise the average log probabilities.
    #     """
    #     if self.finetuning_args.use_ref_model:
    #         batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

    #     all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

    #     all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
    #     if self.loss_type in ["ipo", "orpo", "simpo"]:
    #         all_logps = all_logps / valid_length

    #     batch_size = batch["input_ids"].size(0) // 2
    #     chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
    #     chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
    #     chosen_length, _ = valid_length.split(batch_size, dim=0)
    #     return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    # def compute_reference_log_probs(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    #     r"""
    #     Computes log probabilities of the reference model.
    #     """
    #     ref_model = self.ref_model

    #     with torch.no_grad(), ref_context:
    #         reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

    #     return reference_chosen_logps, reference_rejected_logps

    def get_batch_reference_loss_metrics(self, prompts, input_ids, position_ids, mask, labels, tokenizer):
        # 开启eval模式 
        seq = input_ids
        self.ref_model.eval()
        ref_model_compute_res_start = time.time()
        # seq = input_ids  # [batch_size, prompt_response_max_len]
        pad_token_id = tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            logits_buffer_ref_buffer = torch.Tensor(seq.shape[0], seq.shape[1], self.args.model_actor_config.vocab_size).to(self.args.local_rank)
            rel_logits = torch.zeros_like(logits_buffer_ref_buffer)
            # reward_score = torch.zeros_like(reward_buffer)
            # values = torch.zeros_like(values_buffer)

            data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
           
            rel_loss, rel_outputs = self.ref_model.eval_batch(data_iter=data_iter, return_logits=True, mode="eval")
            if self.ref_model.is_last_stage():
                rel_logits, = rel_outputs
                rel_logits = rel_logits.to(torch.float32)

            dist.all_reduce(rel_logits, group=self.ref_model.mpu.get_pipe_parallel_group())

        # reference_chosen_logps, reference_rejected_logps, *_ = self.post_process(input_ids, rel_logits, labels)

        (
            reference_chosen_logps,
            reference_rejected_logps,
            reference_chosen_logits,
            reference_rejected_logits,
            reference_chosen_logps_avg,
        ) = self.post_process(input_ids, rel_logits, labels)

        ref_model_compute_res_end = time.time()
        self.generate_time = ref_model_compute_res_end - ref_model_compute_res_start
  
        # return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
        return {
            "prompt":prompts,
            "input_ids":input_ids,
            "position_ids":position_ids,
            "mask":mask,
            "labels":labels,
            "reference_chosen_logps":reference_chosen_logps,
            "reference_rejected_logps":reference_rejected_logps,
            "reference_chosen_logits":reference_chosen_logits,
            "reference_rejected_logits":reference_rejected_logits,
            "reference_chosen_logps_avg":reference_chosen_logps_avg
        }


    def get_batch_loss_metrics(self, prompts, input_ids, position_ids, mask, labels, step, tokenizer, tokenizer_ref=None):
        self.eval_pp()
        generate_start = time.time()
        seq = input_ids  # [batch_size, prompt_response_max_len]
        generate_end = time.time()
        self.train_pp()
        pad_token_id = tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():

            logits_buffer_actor_buffer = torch.Tensor(seq.shape[0], seq.shape[1], self.args.model_actor_config.vocab_size).to(self.args.local_rank)
            logits_buffer_ref_buffer = torch.Tensor(seq.shape[0], seq.shape[1], self.args.model_actor_config.vocab_size).to(self.args.local_rank)
            # reward_buffer = torch.Tensor(seq.shape[0]).to(self.args.local_rank)
            # values_buffer = torch.Tensor(seq.shape[0], seq.shape[1]).to(self.args.local_rank)

            logits = torch.zeros_like(logits_buffer_actor_buffer)
            rel_logits = torch.zeros_like(logits_buffer_ref_buffer)
            # reward_score = torch.zeros_like(reward_buffer)
            # values = torch.zeros_like(values_buffer)

            data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
            actor_loss, outputs = self.actor_model.eval_batch(data_iter=data_iter, return_logits=True, mode="train")
            if self.actor_model.is_last_stage():
                logits, = outputs
                logits = logits.to(torch.float32)
           
            rel_loss, rel_outputs = self.ref_model.eval_batch(data_iter=data_iter, return_logits=True, mode="eval")
            if self.ref_model.is_last_stage():
                rel_logits, = rel_outputs
                rel_logits = rel_logits.to(torch.float32)

            dist.all_reduce(logits, group=self.actor_model.mpu.get_pipe_parallel_group())
       
            dist.all_reduce(rel_logits, group=self.ref_model.mpu.get_pipe_parallel_group())
        
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.post_process(input_ids, logits, labels)

        reference_chosen_logps, reference_rejected_logps, *_ = self.post_process(input_ids, rel_logits, labels)

        # compute_preference_loss
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "train_"
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics   

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_dpo(self, inputs):
        # 获取reference model结果inputs
        #   "prompt":prompts,
        #     "input_ids":input_ids,
        #     "position_ids":position_ids,
        #     "mask":mask,
        #     "labels":labels,
        #     "reference_chosen_logps":reference_chosen_logps,
        #     "reference_rejected_logps":reference_rejected_logps,
        #     "reference_chosen_logits":reference_chosen_logits,
        #     "reference_rejected_logits":reference_rejected_logits,
        #     "reference_chosen_logps_avg":reference_chosen_logps_avg
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        reference_chosen_logps = inputs["reference_chosen_logps"]
        reference_rejected_logps = inputs["reference_rejected_logps"]

        data_iter_actor = self._gen_data_iter_dpo_actor_loss(input_ids=input_ids, labels=labels, reference_chosen_logps=reference_chosen_logps, reference_rejected_logps=reference_rejected_logps)
        # 切换loss计算函数，计算后反向传播
        self.actor_model.module.loss_fn = dpo_loss_fn
        actor_loss = self.actor_model.train_batch(data_iter=data_iter_actor)



        # 开始训练
        pass


    def train_rlhf2(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            # old rewards   [batchsize, len(seq) - 1]
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
                # advantages, returns  (batchsize, values.shape[-1] - start)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        # batch = {'input_ids': seq, "attention_mask": attention_mask}
        # actor_prob = self.actor_model(**batch, use_cache=False).logits

        # data_iter = self._gen_data_iter_train(mode="train", input_ids=seq)
        data_iter_actor = self._gen_data_iter_ppo_actor_loss(input_ids=seq, start=start, log_probs=log_probs, advantages=advantages, action_mask=action_mask, cliprange=self.cliprange)
        # 切换loss计算函数，计算后反向传播
        self.actor_model.module.loss_fn = actor_loss_fn
        actor_loss = self.actor_model.train_batch(data_iter=data_iter_actor)

        a = 3
        # if self.actor_model.is_last_stage():
        #     actor_prob, = outputs
        #     actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        #     actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
        #                                     log_probs[:, start:], advantages,
        #                                     action_mask[:, start:])

            # if not self.args.align_overflow:
            #     self.actor_model.step()

        # value = self.critic_model.forward_value(**batch,
        #                                         return_value_only=True,
        #                                         use_cache=False)[:, :-1]


        # exit(0)
      
      

        if self.args.critic_use_pp:
            # to do
            data_iter_critic = self._gen_data_iter_ppo_critic_loss(input_ids=seq, start=start, old_values=old_values, returns=returns, 
                                                               action_mask=action_mask,
                                                                cliprange_value=self.cliprange_value, 
                                                                pad_token_id=self.pad_token_id,
                                                                prompt_len=prompts.shape[1])
            self.critic_model.module.loss_fn = critic_loss_fn
            critic_loss, outputs = self.critic_model.train_batch(data_iter=data_iter_critic)
            pass
        else:
            value = self.critic_model.forward_value(input_ids=seq,
                                                     attention_mask=attention_mask,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
            critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
            self.critic_model.backward(critic_loss)
            a = 4

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            # self.actor_model.step()

        a = 3
        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        return actor_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()

    def train_pp(self):
        self.actor_model.module.train()

    def eval(self):
        self.actor_model.eval()
        self.ref_model.eval()

    def eval_pp(self):
        self.actor_model.module.eval()
        self.ref_model.module.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)



class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
