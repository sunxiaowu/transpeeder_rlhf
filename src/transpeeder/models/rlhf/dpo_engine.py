# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import transformers
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.deepspeed import HfDeepSpeedConfig

import deepspeed.comm as dist
from transpeeder.utils import jload
from transpeeder.models.rlhf.ds_utils import get_train_ds_config, get_eval_ds_config, read_ds_config
from transpeeder.models.rlhf.model_utils import create_hf_model, create_critic_model
from transpeeder.models.rlhf.utils import *
from transpeeder.utils import logger_rank0 as logger

# from transpeeder.models.llama2_patching import (
#     replace_llama_attn_with_flash_attn,
#     refine_rope,
# )
# from transpeeder.models.qwen1_5_pipeline_model_sp import get_model, initialize_model_parallel, get_sequence_parallel_group,  \
 # get_sequence_parallel_world_size, get_sequence_parallel_rank, _SEQUENCE_PARALLEL_GROUP
# from transpeeder.models.llama_pipeline_model import get_model, get_reward_model
# from transpeeder.models.qwen1_5_pipeline_model import *
from transpeeder.models.llama_pipeline_model import *

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

# from utils.utils import get_train_ds_config, get_eval_ds_config, read_ds_config
# from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
# from utils.model.model_utils import create_hf_model, create_critic_model
# from utils.utils import get_optimizer_grouped_parameters

# from utils.model.llama_pipeline_model import get_model, get_reward_model
# from utils.model.llama2_patching import(
#     replace_llama_attn_with_flash_attn,
#     refine_rope
# )
# from utils.utils import logger_rank0 as logger

"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

def read_ds_config(config_path):
    config = jload(config_path)
    return config

def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device,
            "pin_memory": True,
        },
        "offload_optimizer": {
            "device": device,
            "pin_memory": True,
        },
        
        # "offload_optimizer": {
        #     "device": "nvme",
        #     "nvme_path": "/local_nvme",
        #     "pin_memory": True,
        #     "buffer_count": 4,
        #     "fast_init": False
        # },
        # "offload_param": {
        #     "device": "nvme",
        #     "nvme_path": "/local_nvme",
        #     "pin_memory": True,
        #     "buffer_count": 5,
        #     "buffer_size": 1e8,
        #     "max_in_cpu": 1e9
        # },
        #  "aio": {
        #     "block_size": 262144,
        #     "queue_depth": 32,
        #     "thread_count": 1,
        #     "single_submit": False,
        #     "overlap_events": True
        # },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count(
            )
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        # "fp16": {
        #     "enabled": True,
        #     "loss_scale_window": 100,
        #     "min_loss_scale":0.25,
        # },
    #     "fp16": {
    #     "enabled": True,
    #     "auto_cast": False,
    #     "loss_scale": 0,
    #     "initial_scale_power": 16,
    #     "loss_scale_window": 1000,
    #     "hysteresis": 2,
    #     "consecutive_hysteresis": False,
    #     "min_loss_scale": 1
    # },
         "bfloat16": {
            "enabled": True,
        },
        "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": True
    },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        # "offload_param": {
        #     "device": "nvme",
        #     "nvme_path": "/local_nvme",
        #     "pin_memory": True,
        #     "buffer_count": 5,
        #     "buffer_size": 1e8,
        #     "max_in_cpu": 1e9
        # },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }





def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedDPOEngine():

    def __init__(self, args, tokenizer_actor=None, 
                 actor_model_name_or_path=None,
                 tokenizer_ref=None, 
                 ref_model_name_or_path=None):
        # pp = args.pipe_parallel_size
        # mp = args.model_parallel_size
        # assert args.world_size % (pp * mp) == 0
        # dp = args.world_size // (pp * mp)
        self.args = args
        # self.num_total_iters = num_total_iters
        self.tokenizer_actor = tokenizer_actor
        self.tokenizer_ref = tokenizer_ref

        if actor_model_name_or_path is None:
            self.actor = None
        else:
            self.actor = self._init_actor_pipeline(
                actor_model_name_or_path=actor_model_name_or_path)
            
   
        if ref_model_name_or_path is None:
            self.ref = None
        else:
            if self.args.use_ref_model:
                # ref 可能会根据新的pair数据的chosen data 做下sft，使得模型事先学习此数据分布
                if ref_model_name_or_path is None:
                    self.ref = self._init_ref_pipeline(
                        actor_model_name_or_path=actor_model_name_or_path)
                else:
                    self.ref = self._init_ref_pipeline(
                        ref_model_name_or_path=ref_model_name_or_path
                    )
            else:
                self.ref = None


    def load_actor_pipeline(self, actor_model_name_or_path):
        logger.info("start load actor pipeline...")
        self.actor = self._init_actor_pipeline(
                actor_model_name_or_path=actor_model_name_or_path)
        logger.info("end ....")

    def load_ref_pipeline(self, ref_model_name_or_path):
        logger.info("start load ref pipeline...")
        self.ref = self._init_ref_pipeline(
                ref_model_name_or_path=ref_model_name_or_path)
        logger.info(" ref pipeline model load finished ....")

    def destory_pipeline(name="ref"):
        if name == "ref":
            # 销毁 ref model pipeline
            pass
        else:
            pass
    
    def _init_actor_pipeline(self, actor_model_name_or_path):
        stime = log_init("Actor")
        # pipline DS Config
        
        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len +
            self.args.max_answer_seq_len,
            enable_tensorboard=self.args.enable_tensorboard,
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_actor")
        # ds_config[
        #     'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        # #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        # ds_config[
        #     'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        #     ) * self.args.gradient_accumulation_steps_actor
        # 更新新配置
        ds_config = dict()
        ds_config.update(read_ds_config(self.args.deepspeed_config))
        activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)


        # Model
        # actor_model = create_hf_model(
        #     model_class=AutoModelForCausalLM,
        #     model_name_or_path=actor_model_name_or_path,
        #     tokenizer=self.tokenizer,
        #     ds_config=ds_config,
        #     disable_dropout=self.args.disable_actor_dropout)
        model_config = transformers.AutoConfig.from_pretrained(actor_model_name_or_path)
        # model_config.attention_dropout=0.001  # 得到不一致的结果
      
        if self.args.use_flash_attn:
        #     logger.info("⚡⚡⚡ enable flash attention")
        #     replace_llama_attn_with_flash_attn()
        #     refine_rope()
        # model_config._flash_attn_2_enabled = True
            model_config._attn_implementation_internal = "flash_attention_2"

        actor_model = get_model(model_config, self.args, activation_checkpointing_config, partition_method="type:ParallelTransformerLayerPipe")


        # LoRA
        # if self.args.actor_lora_dim > 0:
        #     actor_model = convert_linear_layer_to_lora(
        #         actor_model, self.args.actor_lora_module_name,
        #         self.args.actor_lora_dim)
        #     if self.args.only_optimize_lora:
        #         actor_model = only_optimize_lora_parameters(actor_model)
        #         actor_model = make_model_gradient_checkpointing_compatible(
        #             actor_model)

        # Optimizer
        # AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        # optim_params = get_optimizer_grouped_parameters(
        #     actor_model, self.args.actor_weight_decay,
        #     self.args.actor_lora_learning_rate)
        # optim = AdamOptimizer(optim_params,
        #                       lr=self.args.actor_learning_rate,
        #                       betas=(0.9, 0.95))

        # LR Scheduler
        # lr_scheduler = get_scheduler(
        #     name=self.args.lr_scheduler_type,
        #     optimizer=optim,
        #     num_warmup_steps=self.args.num_warmup_steps,
        #     num_training_steps=self.num_total_iters,
        # )

        # DeepSpeed Engine 
        #TODO: move enable_hybrid_engine and pin_parameters to ds_config
        # actor_engine, *_ = deepspeed.initialize(model=actor_model,
        #                                         optimizer=optim,
        #                                         lr_scheduler=lr_scheduler,
        #                                         config=ds_config)
        actor_engine, _, _, _ = deepspeed.initialize(
        self.args,
        model=actor_model,
        model_parameters=[p for p in actor_model.parameters() if p.requires_grad],
    )
        
        log_init("Actor", stime=stime) 
        actor_engine.load_checkpoint(actor_model_name_or_path,
                            load_module_only=True,
                            load_module_strict=False,
                            load_optimizer_states=False,
                            load_lr_scheduler_states=False,)
        # 记录模型配置参数
        actor_engine.config = model_config
        return actor_engine

    def _init_ref_pipeline(self, ref_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        # zero_stage = self.args.actor_zero_stage
        # if zero_stage != 3:
        #     # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
        #     zero_stage = 0
        
        # ds_config = get_eval_ds_config(self.args.offload_reference_model,
        #                                zero_stage)
        ds_config = dict()
        ds_config.update(read_ds_config(self.args.deepspeed_config))
        activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)
        # ds_config[
        #     'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        # #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        # ds_config[
        #     'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
        #     ) * self.args.gradient_accumulation_steps_actor

        model_config = transformers.AutoConfig.from_pretrained(ref_model_name_or_path)

        if self.args.use_flash_attn:
            logger.info("⚡⚡⚡ enable flash attention")
            # replace_llama_attn_with_flash_attn()
            # refine_rope()
            model_config._attn_implementation_internal = "flash_attention_2"

        # ref_model = create_hf_model(AutoModelForCausalLM,
        #                             actor_model_name_or_path, self.tokenizer,
        #                             ds_config)
        ref_model = get_model(model_config, self.args, activation_checkpointing_config, partition_method="type:ParallelTransformerLayerPipe")
        
        ref_engine, _, _, _ = deepspeed.initialize(
        self.args,
        model=ref_model,
        model_parameters=[p for p in ref_model.parameters() if p.requires_grad],
    )
        log_init("Ref", stime=stime)

        ref_engine.load_checkpoint(ref_model_name_or_path,
                            load_module_only=True,
                            load_module_strict=False,
                            load_optimizer_states=False,
                            load_lr_scheduler_states=False,)
        ref_engine.config = model_config
        return ref_engine
