# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
from transpeeder.utils import jload

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

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


