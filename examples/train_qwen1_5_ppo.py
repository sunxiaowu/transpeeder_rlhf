import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import (
    SchedulerType,
    default_data_collator,
)

import torch
import torch.distributed
import transformers
import numpy as np
import deepspeed

# from transpeeder.models.llama_pipeline_model import get_model
# from transpeeder.models.llama2_patching import (
#     replace_llama_attn_with_flash_attn,
#     # refine_rope,
# )
from transpeeder.models.qwen1_5_pipeline_model import get_model
# from transpeeder.models.qwen1_5_pipeline_model_sp import get_model, initialize_model_parallel, get_sequence_parallel_group,  \
# get_sequence_parallel_world_size, get_sequence_parallel_rank, _SEQUENCE_PARALLEL_GROUP
#from transpeeder.models.llama_pipeline_model import get_model, get_reward_model
from transpeeder.feeder_dp_refine import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
    make_prompt_template_dataloader,
    make_prompt_template_dataloader_rlhf_data_info,
    make_prompt_template_dataloader_rlhf_transpeeder,
    make_prompt_template_dataloader_dpo_transpeeder
)
from transpeeder.models.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from transpeeder.models.rlhf.ppo_trainer import DeepSpeedPPOTrainer

from transpeeder.utils import jload
from transpeeder.utils import logger_rank0 as logger
from transpeeder.models.rlhf.utils import  print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    save_zero_three_model, moving_average, save_zero_three_model, load_hf_tokenizer
from transpeeder.utilsTool.data.data_utils import MiniDataset
from transpeeder.perf import print_throughput_step3

warnings.filterwarnings("ignore")

@dataclass
class TrainerArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)
    use_sp: Optional[bool] = field(default=False)
    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    sequence_parllel_size: int = field(default=-1)
    sequence_parallel_rank: int = field(default=-1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)
    ########## RLHF config ##############
    # actor and ref model config
    offload: bool = field(default=False)
    actor_zero_stage: int = field(default=2)
    enable_hybrid_engine: bool = field(default=False)
    inference_tp_size: int = field(default=1)
    release_inference_cache: bool = field(default=False)
    unpin_actor_parameters: bool = field(default=False)
    tp_gather_partition_size: int = field(default=8)
    max_prompt_seq_len: int = field(default=128)
    max_answer_seq_len: int = field(default=128)
    enable_tensorboard: bool = field(default=False)
    enable_mixed_precision_lora: bool = field(default=False)
    tensorboard_path: str = field(default="step3_tensorboard")
    tb_name: str = field(default="step3_actor")
    offload_reference_model: bool = field(default=False)
    zero_stage_ref_model: int = field(default=3)
    enable_ema: bool = field(default=False)
    actor_model_name_or_path: str = field(default=None)
    end_of_conversation_token: str = field(default="<|endoftext|>")

    # reward and critic model config
    # rm_data_path: str = field(default="/platform_tech/sunshuanglong/models/rm-static")
    rm_data_path: str = field(default="/home/sunshuanglong/transpeeder/data/alpaca_gpt_data_zh_choose_539.jsonl")
    rm_output_path: str = field(default="/tmp/data_files_sun")
    data_split:str = field(default="2,4,4")

    critic_use_pp: bool = field(default=False) 
    critic_gradient_checkpointing: bool = field(default=False)
    critic_offload: bool = field(default=False)
    critic_zero_stage: int = field(default=2)
    per_device_training_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    num_padding_at_beginning: int = field(default=1, metadata={"help":"OPT model has a fixed number (1) of \
        padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."})
    disable_critic_dropout: bool = field(default=True)
    critic_weight_decay: float = field(default=0., metadata={"help":"Weight decay to use."})
    critic_learning_rate: float = field(default=5e-6, metadata={"help":"Initial learning rate (after the potential warmup period) to use."})
    critic_lora_learning_rate: float = field(default=5e-4, metadata={"help":"Initial critic LoRA learning rate (after the potential warmup period) to use."})
    critic_lr_scheduler_type: SchedulerType = field(default="cosine", metadata={"help":"linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup"})
    critic_num_warmup_steps: int = field(default=100, metadata={"help":"Number of steps for the warmup in the lr scheduler."})
    critic_num_total_iters: int = field(default=500)
    critic_model_name_or_path: str = field(default=None)
    deepspeed_config_critic: Optional[str] = field(default=None)

    # rlhf
    rlhf_num_train_epochs: int = field(default=1)
    print_answers: bool = field(default=False)
    generate_batch_size: int = field(default=2)
    ppo_epochs: int = field(default=1)
    align_overflow: bool = field(default=False, metadata={"help":"Align loss scale overflow between actor and critic"})
    enable_test_mode: bool = field(default=False, metadata={"help":'Enable a testing mode that terminates training based on args.test_stop_step'})
    test_stop_step: int = field(default=0)

    # ori transpeeder
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    input_format: Literal['raw', 'tokenized'] = 'raw'
    input_format_rlhf: Literal["data_info", "transpeeder"]= "transpeeder"
    mode: Literal['sft', 'pretrain', 'dialog'] = 'sft'
    num_workers: int = field(default=1)

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    save_epoch_each: int = field(default=1)
    log_steps: int = field(default=1)

    resume_step: int = field(default=-1)
    resume_ckpt: str = field(default="llama-7B-init-test-ckpt")
    ntk : Optional[bool] = field(default=False)


def read_ds_config(config_path):
    config = jload(config_path)
    return config

def main():
    parser = transformers.HfArgumentParser(TrainerArguments)
    args, = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = torch.distributed.get_world_size()
    args.global_rank = torch.distributed.get_rank()
    device = torch.device("cuda", args.local_rank)

    ds_config = read_ds_config(args.deepspeed_config)
    # args.num_workers = 2 * args.world_size // args.pipe_parallel_size // args.model_parallel_size
    args.num_workers = 1
    args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    # activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    tokenizer_actor = transformers.AutoTokenizer.from_pretrained(
        args.actor_model_name_or_path,
        model_max_length=args.max_seq_len,
        padding_side="right",
        # use_fast=False,
    )

    tokenizer_critic = transformers.AutoTokenizer.from_pretrained(
        args.critic_model_name_or_path,
        model_max_length=args.max_seq_len,
        padding_side="right",
        # use_fast=False,
    )

    args.model_actor_config = transformers.AutoConfig.from_pretrained(args.actor_model_name_or_path)
    args.model_critic_config = transformers.AutoConfig.from_pretrained(args.critic_model_name_or_path)
    # model_config = transformers.AutoConfig.from_pretrained(args.actor_model_name_or_path)


    # if args.use_flash_attn:
    #     logger.info("⚡⚡⚡ enable flash attention.")
    #     replace_llama_attn_with_flash_attn()
    #     # refine_rope()
    #     print()

    # if args.ntk:
    #     rope_scaling = {
    #         "type": "dynamic",
    #         "factor": 2,
    #     }
    #     model_config.rope_scaling = rope_scaling
    #     logger.info(f"Turn on dynamic rope for llama2")

    # if args.use_sp:
    #     _SEQUENCE_PARALLEL_GROUP_VAR = initialize_model_parallel(args)
    #     assert _SEQUENCE_PARALLEL_GROUP_VAR is not None
    #     args.sequence_parallel_rank = get_sequence_parallel_rank()
    #     args.sequence_parallel_world_size = get_sequence_parallel_world_size()
    #     logger.info(f"Turn on sequence parallel")
    args.sequence_parallel_rank = 1
    args.sequence_parallel_world_size = 1


        
    # dataset
    # dataloader_maker = make_tokenized_dataloader if args.input_format == 'tokenized' else make_prompt_dataloader
    # train_dataloader = dataloader_maker(tokenizer=tokenizer, data_args=args)

    # pipeline model
    # model = get_model(model_config, args, activation_checkpointing_config, partition_method="type:ParallelTransformerLayerPipe")

    # engine, _, _, _ = deepspeed.initialize(
    #     args,
    #     model=model,
    #     model_parameters=[p for p in model.parameters() if p.requires_grad],
    # )

    rlhf_engine = DeepSpeedRLHFEngine(      
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer_actor=tokenizer_actor,
        tokenizer_critic=tokenizer_critic,
        args=args)
    # rlhf_engine = None
    # a = 3
    # exit(0)

    # dataset
    # dataloader_maker = make_tokenized_dataloader if args.input_format == 'tokenized' else make_prompt_dataloader
    dataloader_maker = make_prompt_template_dataloader_rlhf_transpeeder if args.input_format_rlhf == 'transpeeder' else make_prompt_template_dataloader_rlhf_data_info
    prompt_train_dataloader = dataloader_maker(tokenizer_actor=tokenizer_actor, tokenizer_critic=tokenizer_critic, data_args=args, engine=rlhf_engine)

    trainer = DeepSpeedPPOTrainer(rlhf_engine, args)

    # a = 3
    # exit(0)
    # use `convert2ckpt.py`
    # if args.resume_step < 0:
    #     engine.load_checkpoint(args.init_ckpt,
    #                         load_module_strict=False,
    #                         load_module_only=True,
    #                         load_optimizer_states=False,
    #                         load_lr_scheduler_states=False,
    #     )
    # else:
    #     engine.load_checkpoint(args.resume_ckpt)

    exp_mini_dataset = MiniDataset(args.generate_batch_size,
                                   args.per_device_training_batch_size)

    logger.info(f"***** Running training *****")
    non_overflow_step_count = 0
    for epoch in range(args.rlhf_num_train_epochs):
        logger.info(
            f"Beginning of Epoch {epoch+1}/{args.rlhf_num_train_epochs}, Total Generation Batches {len(prompt_train_dataloader[0])}",
            args.global_rank)
        for step, batch_prompt in enumerate(prompt_train_dataloader[0]):
            batch_prompt = to_device(batch_prompt, device)
            prompts = batch_prompt['prompt']
            length = prompts.size(-1)
            if length > args.max_prompt_seq_len:
                prompts = prompts[:, length - args.max_prompt_seq_len:]
                raise ValueError("Prompt length is too long")

            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              batch_prompt["prompt_critic"],
                                              batch_prompt["prompt_att_mask_critic"],
                                              step, tokenizer_actor, tokenizer_critic)

            training_start = time.time()
    #         # if batch_unsupervised is not None:
    #         #     batch_unsupervised = to_device(batch_unsupervised, device)
    #         #     unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
    #         # else:
    #         #     unsup_dataset = unsup_mini_dataset.add(
    #         #         [[None] * args.per_device_generation_batch_size])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                for ppo_ep in range(args.ppo_epochs):
                    for i, exp_data in enumerate(exp_dataset):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        a = 3
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                    #     if unsupervised_training_enabled:
                    #         unsup_loss = trainer.train_unsupervised(
                    #             unsup_data, args.unsup_coef)      
                    #         unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        # if args.enable_ema:
                        #     moving_average(rlhf_engine.actor,
                        #                    rlhf_engine.actor_ema,
                        #                    zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    # random.shuffle(unsup_dataset)

                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generate_batch_size  # it is an approximation, we did not include, e.g., rw forward time etc

                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | PPO Epoch: {ppo_ep+1} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter}',
                    args.global_rank)
                # print_throughput_step3(rlhf_engine.actor.module,
                #                        rlhf_engine.critic, args, e2e_time,
                #                        trainer.generate_time, training_time,
                #                        args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

                # if args.enable_tensorboard and torch.distributed.get_rank(
                # ) == 0:
                #     writer.add_scalar('reward',
                #                       average_reward / inner_iter,
                #                       global_step=step)
                #     writer.add_scalar('actor_loss',
                #                       actor_loss,
                #                       global_step=step)
                #     writer.add_scalar('actor_loss_sum',
                #                       actor_loss_sum,
                #                       global_step=step)
                #     writer.add_scalar('critic_loss',
                #                       critic_loss,
                #                       global_step=step)
                #     writer.add_scalar('critic_loss_sum',
                #                       critic_loss_sum,
                #                       global_step=step)
                #     writer.flush()
            # config中配置
            # if args.actor_gradient_checkpointing:
            #     rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow, critic_overflow = trainer.get_overflow()

            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break
        # 一个完成的epoch保存一次
        if args.rlhf_num_train_epochs % args.save_epoch_each == 0: 
            print_rank_0('saving model ...')
            logger.info(f"Saving at epoch: {epoch}")
            rlhf_engine.actor.save_checkpoint(args.output_dir)
            if torch.distributed.get_rank() == 0:
                   save_hf_format(rlhf_engine.critic,
                           tokenizer_critic,
                           args,
                           sub_folder='critic')
            if args.critic_zero_stage == 3:
                save_zero_three_model(rlhf_engine.critic,
                                    global_rank=args.global_rank,
                                    save_dir=os.path.join(
                                        args.output_dir, 'critic'),
                                    zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
