import os
import math
import transformers
import torch
import torch.distributed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler
)

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam

from transpeeder.feeder_dp_refine import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
    make_prompt_template_dataloader,
    make_prompt_template_dataloader_rlhf_data_info,
    make_prompt_template_dataloader_rlhf_transpeeder
)


from transpeeder.utils import jload
from transpeeder.utils import logger_rank0 as logger
from transpeeder.models.rlhf.utils import to_device
from transpeeder.utilsTool.data.data_utils import DataCollatorReward, create_prompt_dataset
from transpeeder.models.rlhf.ds_utils import get_train_ds_config, get_eval_ds_config, read_ds_config
from transpeeder.models.rlhf.model_utils import create_hf_model, create_critic_model
from transpeeder.models.rlhf.utils import *
from transpeeder.models.rlhf.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, \
    only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


warnings.filterwarnings("ignore")

@dataclass
class TrainerArguments:
    # reward args
    data_path: str = field(default="/home/sunshuanglong/transpeeder/data/dpo_data.jsonl")
    data_split: str = field(default="2,4,4")
    data_output_path: str = field(default="~/tmp_data/reward_debug")
    model_name_or_path: str = field(default="/platform_tech/models/Qwen1.5-0.5B-Chat")
    data_reload: bool = field(default=False)
    num_padding_at_beginning: int = field(default="1")

    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=5e-5, metadata={"help":"learning rate"})
    max_seq_len:int = field(default=512)
    weight_decay: float = field(default=0.)
    num_train_epochs: int  = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    lr_scheduler_type: SchedulerType = field(default="cosine",  \
                    metadata={"help":"linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup"})
    num_warmup_steps: int = field(default=0)
    output_dir: str = field(default="./output_reward")
    seed: int = field(default=1234)
    local_rank: int = field(default=-1)
    gradient_checkpointing: bool = field(default=False)
    disable_dropout: bool = field(default=True)
    offload: bool = field(default=False)
    zero_stage: int = field(default=1)
    # lora
    lora_dim: int = field(default=0)
    lora_module_name: str = field(default="decoder.layers.")
    only_optimize_lora: bool = field(default=False)
    lora_learning_rate: float = field(default=5e-4) 

    # eval
    eval_interval:int = field(default=0)
    eval_iters: int = field(default=100)
    compute_fp32_loss: bool = field(default=True)

    # log
    enable_tensorboard: bool = field(default=False)
    tensorboard_path: str = field(default="step2_tensorboard")


def read_ds_config(config_path):
    config = jload(config_path)
    return config

def main():
    parser = transformers.HfArgumentParser(TrainerArguments)
    args, = parser.parse_args_into_dataclasses()
    args.local_rank = int(os.environ['LOCAL_RANK'])

    print(args.local_rank)
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        # deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    print(f"global rank:{args.global_rank}")

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    print("seed set finished")
    print("barrier start ...")
    # if args.local_rank not in [0, 1]:
    torch.distributed.barrier()
    print("barrier finished")
    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    print("create tokenizer finished...")
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout,

                                   compute_fp32_loss=args.compute_fp32_loss)
    print("load model finished")
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len, reload=args.data_reload)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        # model.to(device)
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        print(f"eval data len:{len(eval_dataloader)}")
        for step, batch in enumerate(tqdm(eval_dataloader, desc="eval ...")):
            batch = to_device(batch, device)
            with torch.no_grad():
                # print("%" * 30 + "\n")
                # print(f"batch:{batch['input_ids'].shape}\n")
                # print(f"model:{model.device}")
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            # if step == 5:  # For faster evaluation and debugging
            #     break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        return scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            print_rank_0(
                f"Epoch {epoch+1}/{args.num_train_epochs} with step:{step} loss {mean_loss/(step+1)}",
                args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()