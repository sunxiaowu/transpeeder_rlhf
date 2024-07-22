# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType

import torch
from deepspeed import comm as dist
from types import MethodType
from typing import Optional, Literal, List, Tuple
from tqdm import tqdm

from deepspeed.runtime.pipe.engine import PipelineEngine
from transformers import AutoTokenizer


class PipelineEnginePack(PipelineEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    def generate(
        self,
        prompt_tokens: List[List[int]],  # 输入的提示
        max_gen_len: int,  # 最大生成长度
        temperature: float = 0.6,  # 影响生成文本的随机性
        top_p: float = 0.9,  # 用于决定采样过程中保留的 token 集合的概率阈值
        logprobs: bool = False,  # 是否返回每个 token 的对数概率
        echo: bool = False,  # 是否返回输入的提示
        tokenizer:AutoTokenizer = None
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    # ---------------------------初始化长度为 total_len tokens张量，并填充 pad_id----------------------------------
        # text = ["你好，你叫什么名字", "你好，你是机器人吗"]
        # inputs = tokenizer(text,
        #             max_length=512,
        #             padding="max_length",
        #             truncation=True,
        #             return_tensors="pt")
        # # print(f"inputs:{inputs}")
        # prompt_tokens = inputs["input_ids"].to(self.device)
        # prompt_tokens = torch.load("/platform_tech/sunshuanglong/DeepSpeedExamples/tensor_ji.pt")
        # text = ["hello, what's your name", "what is your name, are you a robot?"]
        # tokenizer.padding_side = "left"
        # inputs = tokenizer(text,
        #             max_length=512,
        #             padding="max_length",
        #             truncation=True,
        #             return_tensors="pt")
        # tokenizer.padding_side="left"
        # # inputs = tokenizer(text, return_tensors="pt")
        #     # print(f"inputs:{inputs}")
        # prompt_tokens = inputs["input_ids"].to(self.device)
        params = self.module.model_config
        bsz = len(prompt_tokens)
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
     
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_sequence_length
        total_len = min(params.max_sequence_length, max_gen_len + max_prompt_len)

        pad_id = tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        # 将prompt_tokens中的token复制到tokens张量中。
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        if logprobs:
            # 创建一个与tokens相同形状的token_logprobs张量，并用0填充
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id
        # -------------------------------------------------------------
        for cur_pos in tqdm(range(min_prompt_len, total_len)):
            # 调用模型的forward方法获取logits
            # logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # create eval dataloader
            
            # eval_dataloader = iter(tokens[:, prev_pos:cur_pos])
            # input_ids = copy.deepcopy(tokens[:, :cur_pos])
            # input_ids = tokens[:, :512]
            # input_ids = tokens[:, :512]
            # create data format
            # input_ids = torch.load("/platform_tech/sunshuanglong/DeepSpeedExamples/tensor_ji.pt")
            input_ids = tokens[:, :cur_pos]
            iter_data = [(
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
                # attention_mask

            ),
            (
                 input_ids
            )
            )]
            generate_prompt_iter = iter(deepspeed.utils.RepeatingLoader(iter_data))
            self.module.loss_fn = loss_fn_generate
            eval_loss, outputs = self.eval_batch(data_iter=generate_prompt_iter, return_logits=True)
            # 计算next_token
            next_token = torch.tensor([0] * input_ids.shape[0]).to(self.device)
            if self.is_last_stage():
                logits, = outputs
                if logprobs:
                    # 计算token level的logprobs
                    token_logprobs[:, prev_pos + 1: cur_pos + 1] = -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1: cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                # 根据温度参数和top_p参数对logits进行softmax和采样，得到下一个token
                if temperature > 0:
                    # sample_top_p函数对probs进行采样
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    # 将logits中概率最大的token作为下一个token。
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
            # next_token = next_token.cpu()
            # dist.barrier()
            dist.all_reduce(next_token, group=self.mpu.get_pipe_parallel_group())
            # tokens张量更新,多进程间变量同步
            next_token = next_token.to(self.device)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                     next_token == tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            # dist.barrier()
            # 检查是否已经生成了所有的eos token，如果是则停止生成
            if all(eos_reached):
                break
           
        if logprobs:
            # token_logprobs列表化
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            # 对于 tokens 张量中的每一行（即每一个生成的序列），如果 echo 参数为假，则去掉提示部分
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            # 存在结束标记，则去掉结束标记之后的部分
            # if tokenizer.eos_token_id in toks:
            #     eos_idx = toks.index(tokenizer.eos_token_id)
            #     toks = toks[:eos_idx]
            #     probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        # 返回生成的tokens和对数概率（如果logprobs参数为真）
        out_tokens = torch.tensor(out_tokens, device=self.device)
        return (out_tokens, out_logprobs if logprobs else None)