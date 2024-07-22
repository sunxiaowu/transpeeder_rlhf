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
# from transpeeder.models.qwen1_5_pipeline_model import get_model
# from transpeeder.models.qwen1_5_pipeline_model_sp import get_model, initialize_model_parallel, get_sequence_parallel_group,  \
#  get_sequence_parallel_world_size, get_sequence_parallel_rank, _SEQUENCE_PARALLEL_GROUP
from transpeeder.models.llama_pipeline_model import get_model, get_reward_model
from transpeeder.feeder_dp_refine import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
    make_prompt_template_dataloader,
    make_prompt_template_dataloader_rlhf_data_info,
    make_prompt_template_dataloader_rlhf_transpeeder
)
from transpeeder.models.rlhf.rlhf_engine import DeepSpeedRLHFEngine

from transpeeder.utils import jload
from transpeeder.utils import logger_rank0 as logger

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
    offload: bool = field(default=True)
    actor_zero_stage: int = field(default=2)
    enable_hybrid_engine: bool = field(default=False)
    inference_tp_size: int = field(default=1)
    release_inference_cache: bool = field(default=False)
    unpin_actor_parameters: bool = field(default=False)
    tp_gather_partition_size: int = field(default=8)
    max_prompt_seq_len: int = field(default=64)
    max_answer_seq_len: int = field(default=32)
    enable_tensorboard: bool = field(default=False)
    enable_mixed_precision_lora: bool = field(default=False)
    tensorboard_path: str = field(default="step3_tensorboard")
    tb_name: str = field(default="step3_actor")
    offload_reference_model: bool = field(default=False)
    zero_stage_ref_model: int = field(default=3)
    enable_ema: bool = field(default=False)
    actor_model_name_or_path: str = field(default=None)

    # reward and critic model config
    rm_data_path: str = field(default="/platform_tech/sunshuanglong/models/rm-static")
    rm_output_path: str = field(default="/tmp/data_files_sun")
    data_split:str = field(default="2,4,4")

    critic_use_pp: bool = field(default=False) 
    critic_gradient_checkpointing: bool = field(default=False)
    critic_offload: bool = field(default=True)
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

    ds_config = read_ds_config(args.deepspeed_config)
    # args.num_workers = 2 * args.world_size // args.pipe_parallel_size // args.model_parallel_size
    args.num_workers = 1
    args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.init_ckpt,
        model_max_length=args.max_seq_len,
        padding_side="right",
        # use_fast=False,
    )
    model_config = transformers.AutoConfig.from_pretrained(args.init_ckpt)

    # if args.use_flash_attn:
    #     logger.info("⚡⚡⚡ enable flash attention.")
    #     replace_llama_attn_with_flash_attn()
    #     # refine_rope()
    #     print()

    if args.ntk:
        rope_scaling = {
            "type": "dynamic",
            "factor": 2,
        }
        model_config.rope_scaling = rope_scaling
        logger.info(f"Turn on dynamic rope for llama2")

    # if args.use_sp:
    #     _SEQUENCE_PARALLEL_GROUP_VAR = initialize_model_parallel(args)
    #     assert _SEQUENCE_PARALLEL_GROUP_VAR is not None
    #     args.sequence_parallel_rank = get_sequence_parallel_rank()
    #     args.sequence_parallel_world_size = get_sequence_parallel_world_size()
    #     logger.info(f"Turn on sequence parallel")


        
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

    # rlhf_engine = DeepSpeedRLHFEngine(
    #     actor_model_name_or_path=args.actor_model_name_or_path,
    #     critic_model_name_or_path=args.critic_model_name_or_path,
    #     tokenizer=tokenizer,
    #     # num_total_iters=num_total_iters,
    #     args=args)
    rlhf_engine = None
    # a = 3
    # exit(0)

    # dataset
    # dataloader_maker = make_tokenized_dataloader if args.input_format == 'tokenized' else make_prompt_dataloader
    dataloader_maker = make_prompt_template_dataloader_rlhf_transpeeder if args.input_format_rlhf == 'transpeeder' else make_prompt_template_dataloader_rlhf_data_info
    train_dataloader = dataloader_maker(tokenizer=tokenizer, data_args=args, engine=rlhf_engine)
    
    a = 3
    exit(0)
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
    logger.info(f"***** Running training *****")
    start = time.time()
    for step in range(1, args.train_steps + 1):
        if step <= args.resume_step:
            micro_batch_num = ds_config['train_batch_size'] // ds_config['train_micro_batch_size_per_gpu']
            [next(train_dataloader) for _ in range (micro_batch_num)]
            logger.info(f"Step={step:>6}, skipped.")
            continue
        a = 3
        loss = engine.train_batch(data_iter=train_dataloader)
        if args.local_rank == 0:
            if step % args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / args.log_steps
                logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                start = now

        if step % args.eval_steps == 0:
            # TODO eval
            pass

        if step % args.save_steps == 0:
            logger.info(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
