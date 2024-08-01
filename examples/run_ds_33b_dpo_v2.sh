#!/bin/bash
# Train script.
set -eux

OUTPUT=""

if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

export LAUNCHER="...../bin/torchrun \
    --nproc_per_node $N_GPUS \
    --nnodes ${WORKER_NUM} \
    --node_rank $ID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "
# ref推理
export CMD="$LAUNCHER ${WORK_DIR}/train_deepseek_dpo.py \
    --output_dir ${OUTPUT} \
    --actor_model_name_or_path \
    --critic_model_name_or_path \
    --rm_data_path ../dpo_data.jsonl \
    --max_prompt_seq_len 8192 \
    --exp_train_dataset_path ../dpo_data.pt \
    --ref_only true \
    --actor_gradient_accumulation_steps 4 \
    --generate_batch_size 8 \
    --loss_type sigmoid \
    --use_ref_model true \
    --max_seq_len 8192 \
    --train_steps 150 \
    --eval_steps 10 \
    --save_steps 30 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --ntk false \
    --use_flash_attn true \
    --mode sft \
    --deepspeed_config ${WORK_DIR}/configs/ds_config_zero1_33b_dpo.json"

$CMD

# act训练
export CMD="$LAUNCHER ${WORK_DIR}/train_deepseek_dpo.py \
    --output_dir ${OUTPUT} \
    --actor_model_name_or_path \
    --critic_model_name_or_path \
    --rm_data_path ../dpo_data.jsonl \
    --max_prompt_seq_len 8192 \
    --exp_train_dataset_path ../dpo_data.pt \
    --ref_only false \
    --actor_gradient_accumulation_steps 4 \
    --generate_batch_size 8 \
    --loss_type sigmoid \
    --use_ref_model true \
    --max_seq_len 8192 \
    --train_steps 150 \
    --eval_steps 10 \
    --save_steps 30 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --ntk false \
    --use_flash_attn true \
    --mode dpo \
    --deepspeed_config ${WORK_DIR}/configs/ds_config_zero1_33b_dpo.json"

$CMD
