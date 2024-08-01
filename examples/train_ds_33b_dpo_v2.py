import os
import time
import random
import torch
import torch.distributed
import transformers
import numpy as np
import deepspeed
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import (
    SchedulerType
)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from transpeeder.models.llama_pipeline_model import get_model
# from transpeeder.models.llama2_patching import (
#     replace_llama_attn_with_flash_attn,
#     # refine_rope,
# )
# from transpeeder.models.qwen1_5_pipeline_model import get_model
# from transpeeder.models.qwen1_5_pipeline_model_sp import get_model, initialize_model_parallel, get_sequence_parallel_group,  \
# get_sequence_parallel_world_size, get_sequence_parallel_rank, _SEQUENCE_PARALLEL_GROUP
# from transpeeder.models.llama_pipeline_model import get_model, get_reward_model
from transpeeder.feeder_dp_refine import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
    make_prompt_template_dataloader,
    make_prompt_template_dataloader_rlhf_data_info,
    make_prompt_template_dataloader_dpo_transpeeder,
    make_prompt_template_dataloader_dpo_step2_transpeeder,
    make_prompt_template_dataloader_rlhf_transpeeder,
    PromptTemplateDatasetStep2DPO,
    make_prompt_template_dataloader_dpo_step2_transpeeder_logits
)
from transpeeder.models.rlhf.dpo_engine import DeepSpeedDPOEngine
from transpeeder.models.rlhf.ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedDPOTrainer

from transpeeder.utils import jload
from transpeeder.utils import logger_rank0 as logger
from transpeeder.models.rlhf.utils import  print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    save_zero_three_model, moving_average, save_zero_three_model, load_hf_tokenizer
from transpeeder.utilsTool.data.data_utils import MiniDataset
import pickle
# from transpeeder.perf import print_throughput_step3

warnings.filterwarnings("ignore")

@dataclass
class TrainerArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=True)
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
    max_prompt_seq_len: int = field(default=8192)
    max_answer_seq_len: int = field(default=2048)
    enable_tensorboard: bool = field(default=False)
    enable_mixed_precision_lora: bool = field(default=False)
    tensorboard_path: str = field(default="step3_tensorboard")
    tb_name: str = field(default="step3_actor")
    offload_reference_model: bool = field(default=False)
    zero_stage_ref_model: int = field(default=3)
    enable_ema: bool = field(default=False)
    actor_model_name_or_path: str = field(default=None)
    end_of_conversation_token: str = field(default="")

    # reward and critic model config
    # rm_data_path: str = field(default="")
    rm_data_path: str = field(default="")
    rm_output_path: str = field(default="")
    data_split:str = field(default="2,4,4")

    critic_use_pp: bool = field(default=False) 
    critic_gradient_checkpointing: bool = field(default=False)
    critic_offload: bool = field(default=False)
    critic_zero_stage: int = field(default=2)
    per_dpo_training_batch_size: int = field(default=2)
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
    generate_batch_size: int = field(default=8)
    min_batch_size: int  = field(default=2)
    step_num: int = field(default=1)
    ppo_epochs: int = field(default=1)
    align_overflow: bool = field(default=False, metadata={"help":"Align loss scale overflow between actor and critic"})
    enable_test_mode: bool = field(default=False, metadata={"help":'Enable a testing mode that terminates training based on args.test_stop_step'})
    test_stop_step: int = field(default=0)

    # ori transpeeder
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    input_format: Literal['raw', 'tokenized'] = 'raw'
    input_format_rlhf: Literal["data_info", "transpeeder"]= "transpeeder"
    mode: Literal['sft', 'pretrain', 'dialog', 'dpo'] = 'dpo'
    num_workers: int = field(default=1)

    # dpo
    # dpo_epochs: int = field(default=4)
    exp_train_dataset_path: str = field(default="./output")
    ref_only: bool = field(default=False)
    actor_gradient_accumulation_steps: int = field(default=32)
    loss_type: Literal["sigmoid", "robust", "hinge", "ipo", "kto_pair", "bco_pair",'sppo_hard', "nca_pair", "orpo", "simpo"] = 'sigmoid'
    use_ref_model: bool = field(default=True)
    beta: float = field(default=0.1)
    reference_free: bool = field(default=False)
    label_smoothing: float = field(default=0.0)
    ftx_gamma: float = field(default=0.0)
    save_each_epoch: bool = field(default=True)
    simpo_gamma: float= field(default=0.0)


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

    args.model_actor_config = transformers.AutoConfig.from_pretrained(args.actor_model_name_or_path)

    if not args.use_ref_model:
        tokenizer_critic = tokenizer_actor
        args.model_critic_config = args.model_actor_config
    else:
        tokenizer_critic = transformers.AutoTokenizer.from_pretrained(
            args.critic_model_name_or_path,
            model_max_length=args.max_seq_len,
            padding_side="right",
            # use_fast=False,
        )
        args.model_critic_config = transformers.AutoConfig.from_pretrained(args.critic_model_name_or_path)


    rlhf_engine = DeepSpeedDPOEngine(args=args, tokenizer_actor=tokenizer_actor, tokenizer_ref=tokenizer_critic)
    if args.use_ref_model and args.ref_only:
        rlhf_engine.load_ref_pipeline(ref_model_name_or_path=args.critic_model_name_or_path)

    a = 3

    trainer = DeepSpeedDPOTrainer(rlhf_engine, args)

    if args.ref_only:
        # exp_train_dataset = []
        dataloader_maker = make_prompt_template_dataloader_dpo_transpeeder if args.input_format_rlhf == 'transpeeder' else make_prompt_template_dataloader_rlhf_data_info
        prompt_ref_calcu_dataloader = dataloader_maker(tokenizer_actor=tokenizer_actor, tokenizer_ref=tokenizer_actor, data_args=args, engine=rlhf_engine)
        exp_train_dataset = PromptTemplateDatasetStep2DPO(args)
        logger.info(f"***** Running infering *****")
        non_overflow_step_count = 0
        for epoch in range(args.rlhf_num_train_epochs):
            logger.info(
                f"Beginning of Epoch {epoch+1}/{args.rlhf_num_train_epochs}, Total Generation ref logits data {len(prompt_ref_calcu_dataloader[0]) * args.generate_batch_size}",
                args.global_rank)
            for step_0, batch_prompt in enumerate(prompt_ref_calcu_dataloader[0]):
                batch_prompt = to_device(batch_prompt, device)
                for prompt in batch_prompt["prompt"]:
                    length = prompt.size(-1)
                    if length > args.max_prompt_seq_len:
                        prompt = prompt[:, length - args.max_prompt_seq_len:]
                        raise ValueError("Prompt length is too long")

                out = trainer.get_batch_reference_loss_metrics(args, batch_prompt['prompt'],
                                                batch_prompt['input_ids'],
                                                batch_prompt["position_ids"],
                                                batch_prompt["attn_mask"],
                                                batch_prompt["labels"],
                                                tokenizer_critic)
                torch.cuda.empty_cache()
                exp_train_dataset.add(out)
     
            
                print_rank_0(f"ref model calculate step:{step_0}, finished data num:{len(exp_train_dataset)}", args.local_rank)
            # 撤销ref model，加载actor model
            torch.distributed.barrier()
            rlhf_engine.ref.cpu()
            rlhf_engine.ref.reset_activation_shape()
            del rlhf_engine.ref
            torch.cuda.empty_cache()

        if args.local_rank == 0:
            print("saving exp train dataset.....")
            print(type(exp_train_dataset.samples))
            torch.save(exp_train_dataset.samples, args.exp_train_dataset_path)

    else:

        logger.info(f"***** Running training *****")
        print("load actor model ...")
        rlhf_engine.load_actor_pipeline(actor_model_name_or_path=args.actor_model_name_or_path)
        print_rank_0("end actor model ...", rank=args.local_rank)
        trainer.actor_model = rlhf_engine.actor 
        trainer.actor_model._config.gradient_accumulation_steps = args.actor_gradient_accumulation_steps
        # 构造训练的dataloader
        # 其他po用 make_prompt_template_dataloader_dpo_transpeeder
        train_dataloader = make_prompt_template_dataloader_dpo_step2_transpeeder_logits(args.exp_train_dataset_path, data_args=args, engine=rlhf_engine)
        #  start training...
        trainer.train_dpo_final(args, train_dataloader)
    
if __name__ == "__main__":
    main()