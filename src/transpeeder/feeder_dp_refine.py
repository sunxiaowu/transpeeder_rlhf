""" feader.py """

import copy
import json
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
from collections import defaultdict

import torch
import deepspeed
import transformers
from tqdm import tqdm 
from torch.utils.data import Dataset, Subset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

from .utils import is_rank_0
from .utils import logger_rank0 as logger
from .utilsTool.data.data_utils import *
from .utilsTool.data.raw_datasets import *


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PREFIX_BEGIN_TOKEN = "<|prefix_begin|>"
PREFIX_END_TOKEN   = "<|prefix_end|>"
PROMPTER_TOKEN     = "<|prompter|>"
ASSISTANT_TOKEN    = "<|assistant|>"
ENDOFTEXT_TOKEN    = "<|endoftext|>"

PROMPT_FIELD = 'prompt'
PROMPT_ACTOR_FIELD = "prompt_actor"
PROMOPT_CRITIC = "prompt_critic"
OUTPUT_FIELD = 'output'
RESPONSE = 'response'
CHOSEN = 'chosen'
REJIECTED = 'rejected'


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    batch_tokenized = tokenizer(
        strings,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    input_ids = labels = batch_tokenized
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in batch_tokenized
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _tokenize_fn_no_pad(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    batch_tokenized = tokenizer(
        strings,
        return_tensors="pt"
    ).input_ids

    input_ids = labels = batch_tokenized
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in batch_tokenized
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _make_labels(input_ids, tokenizer: transformers.PreTrainedTokenizer, mode: str = "sft", **kwargs):
    if mode in ["sft", "dpo"]:
        assert "source_lens" in kwargs, f"miss parameter: source_lens"
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, kwargs["source_lens"]):
            label[: source_len] = IGNORE_INDEX
        return labels
    elif mode == "pretrain":
        return copy.deepcopy(input_ids)
    elif mode == "dialog":
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=input_ids.dtype)
        # <|assistant|> ... <|endoftext|>
        ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
        ENDOFTEXT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ENDOFTEXT_TOKEN)
        PROMPTER_TOKEN_ID = tokenizer.convert_tokens_to_ids(PROMPTER_TOKEN)
        for input_row, label_row in zip(input_ids, labels):
            begin_indices = torch.nonzero(input_row == ASSISTANT_TOKEN_ID)
            for idx in begin_indices:
                edi = idx + 1
                while edi < len(input_row) and input_row[edi] != ENDOFTEXT_TOKEN_ID:
                    edi += 1
                if edi < len(input_row) and \
                        input_row[edi + 1] != PROMPTER_TOKEN_ID:
                    logger.warning(f'expect {PROMPTER_TOKEN} after {ENDOFTEXT_TOKEN}, get {input_row[edi + 1]}.')
                label_row[idx + 1: edi + 1] = input_row[idx + 1: edi + 1]

        return labels
    else:
        raise ValueError('Unvalid training mode.')


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    samples = [s + t for s, t in zip(sources, targets)]
    samples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (samples, sources)]
    input_ids = samples_tokenized["input_ids"]
    labels = _make_labels(input_ids, tokenizer, mode,
                          source_lens=sources_tokenized["input_ids_lens"])

    # shift
    if mode != "dpo":
        return dict(
            input_ids=[ids[: -1] for ids in input_ids],
            labels=[lbs[1: ]for lbs in labels]
        )
    else:
        return dict(
            input_ids=[ids[: ] for ids in input_ids],
            labels=[lbs[: ]for lbs in labels]
        )

class PromptDataset(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], eos: str = ""):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'

        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt, output = sample[PROMPT_FIELD], sample[OUTPUT_FIELD]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        self.samples.append(dict(
                            prompt=prompt,
                            output=output + eos,
                        ))
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]

class PromptTemplateDataset(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], tokenizer:transformers.PreTrainedTokenizer):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'
        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt, output = sample[PROMPT_FIELD], sample[OUTPUT_FIELD]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        # self.samples.append(dict(
                        #     prompt=prompt,
                        #     output=output + eos,
                        # ))
                        prompt_template = [dict(
                            role="user",
                            content=prompt
                        )]
                        dict_data = dict(
                            prompt=tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True),
                            output=output + '<|im_end|>'
                        )
                        self.samples.append(dict_data)
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]

class PromptTemplateDatasetRLHF(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], tokenizer_actor:transformers.PreTrainedTokenizer, tokenizer_critic:transformers.PreTrainedTokenizer):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'
        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt = sample.get(PROMPT_FIELD, None)
                        output = sample.get(OUTPUT_FIELD, None)
                        chosen = sample.get(CHOSEN, None)
                        rejected = sample.get(REJIECTED, None)
                        # prompt, output, chosen, rejected = sample[PROMPT_FIELD], sample[OUTPUT_FIELD], sample[CHOSEN], sample[REJIECTED]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        # self.samples.append(dict(
                        #     prompt=prompt,
                        #     output=output + eos,
                        # ))
                        prompt_template = [dict(
                            role="user",
                            content=prompt
                        )]
                        dict_data = dict(
                            prompt=tokenizer_actor.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True),
                            prompt_critic=tokenizer_critic.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True)
                            # output=output + '<|im_end|>'
                        )
                        self.samples.append(dict_data)
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')
        self.samples = self.samples[:20]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]

class PromptTemplateDatasetDPO(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], tokenizer_actor:transformers.PreTrainedTokenizer, tokenizer_critic:transformers.PreTrainedTokenizer=None):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'
        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt = sample.get(PROMPT_FIELD, None)
                        output = sample.get(OUTPUT_FIELD, None)
                        chosen = sample.get(CHOSEN, None)
                        rejected = sample.get(REJIECTED, None)
                        # 检查是否有output
                        if output.strip() in [None, ""] and chosen is not None:
                            output = chosen
                        else:
                            continue
                        # prompt, output, chosen, rejected = sample[PROMPT_FIELD], sample[OUTPUT_FIELD], sample[CHOSEN], sample[REJIECTED]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        # self.samples.append(dict(
                        #     prompt=prompt,
                        #     output=output + eos,
                        # ))
                        prompt_template = [dict(
                            role="user",
                            content=prompt
                        )]
                        dict_data = dict(
                            prompt=tokenizer_actor.apply_chat_template([dict(role="user",content=prompt)], tokenize=False, add_generation_prompt=True),
                            output=output + '<|im_end|>',
                            chosen=chosen + '<|im_end|>',
                            rejected=rejected + '<|im_end|>',
                        )
                        self.samples.append(dict_data)
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')
        self.samples = self.samples[:50]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]

class PromptTemplateDatasetStep2DPO(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, args):
        super().__init__()
        self.samples = list()
        self.args = args
    
    def add(self, data):
        len_data = len(data["prompt"])
        if self.args.use_ref_model:
            for idx in range(len_data):
                self.samples.append({
                    "prompt":data["prompt"][idx],
                    "chosen_input_ids":data["input_ids"][idx],
                    "rejected_input_ids":data["input_ids"][idx + len_data],
                    # "chosen_position_ids":data["position_ids"][idx], 
                    # "rejected_position_ids":data["position_ids"][idx + len_data],
                    "chosen_mask_ids": data["mask"][idx],
                    "rejected_mask_ids":data["mask"][idx + len_data],
                    "chosen_labels_ids":data["labels"][idx],
                    "rejected_labels_ids":data["labels"][idx + len_data],
                    "reference_chosen_logps":data["reference_chosen_logps"][idx],
                    "reference_rejected_logps":data["reference_rejected_logps"][idx],
                    # "reference_chosen_logits":data["reference_chosen_logits"][idx],
                    # "reference_rejected_logits":data["reference_rejected_logits"][idx],
                    "reference_chosen_logps_avg":data["reference_chosen_logps_avg"][idx]
                })
        else:
             for idx in range(len_data):
                self.samples.append({
                    "prompt":data["prompt"][idx],
                    "chosen_input_ids":data["input_ids"][idx],
                    "rejected_input_ids":data["input_ids"][idx + len_data],
                    # "chosen_position_ids":data["position_ids"][idx],
                    # "rejected_position_ids":data["position_ids"][idx + len_data],
                    "chosen_mask_ids": data["mask"][idx],
                    "rejected_mask_ids":data["mask"][idx + len_data],
                    "chosen_labels_ids":data["labels"][idx],
                    "rejected_labels_ids":data["labels"][idx + len_data],
                    "reference_chosen_logps":data["reference_chosen_logps"],
                    "reference_rejected_logps":data["reference_rejected_logps"],
                    # "reference_chosen_logits":data["reference_chosen_logits"],
                    # "reference_rejected_logits":data["reference_rejected_logits"],
                    "reference_chosen_logps_avg":data["reference_chosen_logps_avg"]
                })
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]



@dataclass
class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mode: str
    # seq_parallel_rank:int

    @cache
    @staticmethod
    def get_attn_mask(bs, seq_length):
        """
        Get triangular attention mask.
        """
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length     
        )
        # convert to binary
        return mask < 0.5

    @staticmethod
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        seq_length = input_ids.shape[1]
        # sp_range = seq_length // self.sequence_parallel_world_size

        labels = torch.stack(labels)
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        position_ids = DataCollatorForPromptDataset.get_position_ids(input_ids)
        attn_mask = DataCollatorForPromptDataset.get_attn_mask(input_ids.shape[0], input_ids.shape[1])

        input_ids_section = input_ids
        position_ids_section = position_ids
        attn_mask_section =  attn_mask
        labels_section = labels
       
        return (
                (
                    input_ids_section,
                    position_ids_section,
                    attn_mask_section,
                ),
                labels_section
            )

@dataclass
class DataCollatorForPromptDatasetRLHF(object):
    """Collate for supervised fine-tuning."""

    tokenizer_actor: transformers.PreTrainedTokenizer
    tokenizer_critic: transformers.PreTrainedTokenizer
    mode: str
    seq_parallel_rank:int
    sequence_parallel_world_size:int

    @cache
    @staticmethod
    def get_attn_mask(bs, seq_length):
        """
        Get triangular attention mask.
        """
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    @staticmethod
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        sources_critic = [sample[PROMOPT_CRITIC] for sample in samples]
      

        # data_dict = preprocess(sources, targets, self.tokenizer, self.mode)

        # samples = [s + t for s, t in zip(sources, targets)]
        
        samples_tokenized = [_tokenize_fn_no_pad(strings, self.tokenizer_actor) for strings in sources][0]
        input_ids = samples_tokenized["input_ids"]
        
        samples_tokenized_critic = [_tokenize_fn_no_pad(strings, self.tokenizer_critic) for strings in sources_critic][0]
        input_ids_critic = samples_tokenized_critic["input_ids"]
        # input_ids = torch.stack(input_ids)

        # position_ids = DataCollatorForPromptDataset.get_position_ids(input_ids)
        attn_mask = DataCollatorForPromptDataset.get_attn_mask(input_ids.shape[0], input_ids.shape[1])
        attn_mask_critic = DataCollatorForPromptDataset.get_attn_mask(input_ids_critic.shape[0], input_ids_critic.shape[1])
       
        return {
            "prompt": input_ids,
            "prompt_critic":input_ids_critic,
            "prompt_att_mask": attn_mask,
            "prompt_att_mask_critic":attn_mask_critic
        }

@dataclass
class DataCollatorForPromptDatasetSP(object):
    """Collate for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mode: str
    seq_parallel_rank:int
    sequence_parallel_world_size:int

    @cache
    @staticmethod
    def get_attn_mask(bs, seq_length):
        """
        Get triangular attention mask.
        """
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    @staticmethod
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        if targets[0] == '## 致谢评估润色\n\n致谢部分表达了对导师的真切感激之情，但存在一些表述上的模糊和改善空间，特别是在感谢的具体性和简洁性方面。\n\n优化建议:\n1. 明确指出导师具体如何帮助论文写作，例如在论文结构设计、资料查找等方面的具体指导。\n2. 简化表达，去除重复或冗余的词语，使致谢内容更加精炼。\n3. 添加对其他可能参与支持的人或机构的感谢，如同行评审、资金支持机构等，如果适用。\n4. 使用专业且礼貌的语言，避免过于口语化的表达。\n\n下面是优化后的致谢内容:\n\n致谢\n\n在本论文从构想到完成的过程中，我深深感谢我的导师的悉心指导和无私帮助。导师不仅在论文的选题、结构设计、资料收集与分析等方面提供了专业的建议，还针对论文的初稿提出了宝贵的修改意见，特别推荐使用最新数据，极大提升了我的论文的学术质量和实际应用价值。我也要感谢所有参与讨论和提供反馈的同行和朋友，以及任何提供资金支持的机构，他们的支持让我的研究得以顺利进行。\n\n通过这种方式，致谢部分不仅更加具体、恰当，也更加遵守了学术规范和出版要求，同时保持了必要的简洁性。<|im_end|>':
            a = 3

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        seq_length = input_ids.shape[1]
        sp_range = seq_length // self.sequence_parallel_world_size

        labels = torch.stack(labels)
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        # 检测是否开启sp，sequence切分
        if self.seq_parallel_rank != -1:
            sub_seq_start = self.seq_parallel_rank * sp_range
            sub_seq_end = (self.seq_parallel_rank + 1) * sp_range
        else:
            sub_seq_start = self.seq_parallel_rank * sp_range
            sub_seq_end = (self.seq_parallel_rank + 1) * sp_range
        # label 全为-1的话，直接不要这条
        position_ids = DataCollatorForPromptDataset.get_position_ids(input_ids)
        attn_mask = DataCollatorForPromptDataset.get_attn_mask(input_ids.shape[0], input_ids.shape[1])

        input_ids_section = input_ids[:, sub_seq_start:sub_seq_end]
        position_ids_section = position_ids[:, sub_seq_start:sub_seq_end]
        attn_mask_section =  attn_mask[:, sub_seq_start:sub_seq_end]
        labels_section = labels[:, sub_seq_start:sub_seq_end]
        # # a = 3
        # input_ids_section_res = copy.deepcopy(input_ids_section)
        # position_ids_section_res = copy.deepcopy(position_ids_section)
        # label_section_res = copy.deepcopy(labels_section)
        # attn_mask_section_res = copy.deepcopy(attn_mask_section)

        # ignore_idx = list()
        # for idx, (input_id_sub, attn_mask_sub,  position_id_sub, label_sub) in enumerate(zip(input_ids_section, attn_mask_section, position_ids_section, labels_section)):
        #     is_all_ignore = int(label_sub[0].eq(-100).all())
        #     if is_all_ignore == 1:
        #         a = 3
        #         input_ids_section_res[idx] = input_ids[idx][0:sp_range]
        #         position_ids_section_res[idx] = position_ids[idx][0:sp_range]
        #         attn_mask_section_res[idx] = attn_mask[idx][0:sp_range]
        #         label_section_res[idx] = labels[idx][0:sp_range]
        #         if int(label_section_res.eq(-100).all()) == 1:
        #             a = 1

        return (
                (
                    input_ids_section,
                    position_ids_section,
                    attn_mask_section,
                ),
                labels_section
            )


@dataclass
class DataCollatorForPromptDatasetDPO(object):
    """Collate for supervised fine-tuning."""

    tokenizer_actor: transformers.PreTrainedTokenizer
    tokenizer_ref: transformers.PreTrainedTokenizer
    mode: str
    # seq_parallel_rank:int
    # sequence_parallel_world_size:int

    @cache
    @staticmethod
    def get_attn_mask(bs, seq_length):
        """
        Get triangular attention mask.
        """
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    @staticmethod
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]
        chosen = [sample[CHOSEN] for sample in samples]
        rejected = [sample[REJIECTED] for sample in samples]

        # prompt preprocess
        samples_tokenized = [_tokenize_fn_no_pad(strings, self.tokenizer_actor) for strings in sources]
        prompt_ids = [_["input_ids"] for _ in samples_tokenized]

        # chosen data preprocess
        data_dict = preprocess(sources, targets, self.tokenizer_actor, self.mode)
        input_ids = torch.stack(data_dict["input_ids"])
        labels = torch.stack(data_dict["labels"])
   
        # rejected data preprocess
        data_dict_rejected = preprocess(sources, rejected, self.tokenizer_actor, self.mode)
        input_ids_rejected = torch.stack(data_dict_rejected["input_ids"])
        labels_rejected = torch.stack(data_dict_rejected["labels"])

        input_ids = torch.cat((input_ids, input_ids_rejected), dim=0)
        labels = torch.cat((labels, labels_rejected), dim=0)

        # seq_length = input_ids.shape[1]
        # sp_range = seq_length // self.sequence_parallel_world_size

        labels = torch.where(input_ids == self.tokenizer_actor.pad_token_id, IGNORE_INDEX, labels)

        position_ids = DataCollatorForPromptDataset.get_position_ids(input_ids)
        attn_mask = DataCollatorForPromptDataset.get_attn_mask(input_ids.shape[0], input_ids.shape[1])
       
        return {
                "prompt": prompt_ids,
                "input_ids":input_ids,
                "position_ids":position_ids,
                "attn_mask":attn_mask,
                "labels":labels
        }

@dataclass
class DataCollatorForPromptDatasetStep2DPO(object):
    """Collate for supervised fine-tuning."""
    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids = [sample["chosen_input_ids"] for sample in samples]
        rejected_input_ids = [sample["rejected_input_ids"] for sample in samples]
        input_ids = torch.cat((torch.stack(chosen_input_ids), torch.stack(rejected_input_ids)), dim=0)

        chosen_labels_ids = [sample["chosen_labels_ids"] for sample in samples]
        rejected_labels_ids = [sample["rejected_labels_ids"] for sample in samples]
        labels = torch.cat((torch.stack(chosen_labels_ids), torch.stack(rejected_labels_ids)), dim=0)


        reference_chosen_logps = torch.stack([sample["reference_chosen_logps"] for sample in samples])
        reference_rejected_logps = torch.stack([sample["reference_rejected_logps"] for sample in samples])
        a = 3
        return  (
            (      
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids.shape[0], input_ids.shape[1])
            ),
            (
                input_ids,
                labels,
                reference_chosen_logps,
                reference_rejected_logps
            )
            )

class TokenizedDataset(Dataset):
    def __init__(self, data_path: Union[str, Path]):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'

        self.samples = []
        all_files = list(data_path.glob('**/*.pt') if data_path.is_dir() else [data_path])

        for single_file in tqdm(all_files, disable=not is_rank_0()):
            self.samples.extend(torch.load(single_file))

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        return self.samples[index]


@dataclass
class DataCollatorForTokenizedDataset(DataCollatorForPromptDataset):

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([s['input_ids'] for s in samples])
        labels = torch.stack([s['labels'] for s in samples])
        return (
            (
                input_ids,
                self.get_position_ids(input_ids.shape[0], input_ids.shape[1]),
                self.get_attn_mask(input_ids),
            ),
            labels
        )


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_prompt_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptDataset(data_path=data_args.data_path, eos=tokenizer.eos_token)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()
    # print("setting shuffle--True")
    if data_args.local_rank <= 0:
        print("refine dp")
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))


def make_tokenized_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    dataset = TokenizedDataset(data_path=data_args.data_path)
    data_collator = DataCollatorForTokenizedDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))


def make_prompt_template_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptTemplateDataset(data_path=data_args.data_path, tokenizer=tokenizer)
    # data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode, seq_parallel_rank=data_args.sequence_parallel_rank, sequence_parallel_world_size=data_args.sequence_parallel_world_size)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()
    # print("setting shuffle--True")
    print("refine dp")
    
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))

def make_prompt_template_dataloader_rlhf_data_info(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None, train_phase=3) -> Dict:
    prompt_train_dataset, _ = create_prompt_dataset(
        data_args.local_rank, data_args.rm_data_path, data_args.data_split,
        data_args.rm_output_path, train_phase, data_args.seed, tokenizer,
        data_args.max_prompt_seq_len)
    
     # DataLoaders creation:
    data_collator = DataCollatorRLHF(data_args.max_prompt_seq_len,
                                     data_args.inference_tp_size)

    # train_sampler = DistributedSampler(prompt_train_dataset)

    # # TODO add eval dataloader
    # assert val_split is None
    # dataset = PromptTemplateDataset(data_path=data_args.data_path, tokenizer=tokenizer)
    # data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode, seq_parallel_rank=data_args.sequence_parallel_rank, sequence_parallel_world_size=data_args.sequence_parallel_world_size)
    
    g = torch.Generator()
    # print("setting shuffle--True")
    # print("refine dp")
    
    train_sampler = DistributedSampler(prompt_train_dataset,
                    # num_replicas=engine.dp_world_size,
                    # rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(prompt_train_dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return (dataloader, iter(deepspeed.utils.RepeatingLoader(dataloader)))


def make_prompt_template_dataloader_rlhf_transpeeder(tokenizer_actor: transformers.PreTrainedTokenizer, tokenizer_critic: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptTemplateDatasetRLHF(data_path=data_args.rm_data_path, tokenizer_actor=tokenizer_actor, tokenizer_critic=tokenizer_critic)
    data_collator = DataCollatorForPromptDatasetRLHF(tokenizer_actor=tokenizer_actor, tokenizer_critic=tokenizer_critic, mode=data_args.mode, seq_parallel_rank=data_args.sequence_parallel_rank, sequence_parallel_world_size=data_args.sequence_parallel_world_size)

    g = torch.Generator()
    # print("setting shuffle--True")
    print("refine dp")
    
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.actor.dp_world_size,
                    rank=engine.actor.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.generate_batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return (dataloader, iter(deepspeed.utils.RepeatingLoader(dataloader)))

    assert val_split is None
    dataset = PromptTemplateDataset(data_path=data_args.data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode, seq_parallel_rank=data_args.sequence_parallel_rank, sequence_parallel_world_size=data_args.sequence_parallel_world_size)

    g = torch.Generator()
    # print("setting shuffle--True")
    print("refine dp")
    
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))


def make_prompt_template_dataloader_dpo_transpeeder(tokenizer_actor: transformers.PreTrainedTokenizer, tokenizer_ref: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptTemplateDatasetDPO(data_path=data_args.rm_data_path, tokenizer_actor=tokenizer_actor)
    data_collator = DataCollatorForPromptDatasetDPO(tokenizer_actor=tokenizer_actor, tokenizer_ref=tokenizer_ref, mode=data_args.mode)

    g = torch.Generator()
    # print("setting shuffle--True")
    print("refine dp")
    if data_args.use_ref_model:
        num_replicas = engine.ref.dp_world_size
        rank = engine.ref.mpu.get_data_parallel_rank()
    else:
        num_replicas = engine.actor.dp_world_size
        rank = engine.actor.mpu.get_data_parallel_rank()

    train_sampler = DistributedSampler(dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.generate_batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return (dataloader, iter(deepspeed.utils.RepeatingLoader(dataloader)))

def make_prompt_template_dataloader_dpo_step2_transpeeder(dataset, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    # dataset = PromptTemplateDatasetDPO(data_path=data_args.rm_data_path, tokenizer_actor=tokenizer_actor)
    data_collator = DataCollatorForPromptDatasetStep2DPO()

    g = torch.Generator()
    # print("setting shuffle--True")
    print("refine dp")
    
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.actor.dp_world_size,
                    rank=engine.actor.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            # shuffle=True,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))
