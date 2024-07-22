#!/bin/bash
# Usage: srun train_slurm.sh

set -eux

# 任务名
name=intros-v1-polish-d1107

export MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
export MASTER_PORT=23858
export OMP_NUM_THREADS=8
#export CUDA_LAUNCH_BLOCKING=1
export WORK_DIR=`pwd`

# 日志路径
LOG_PATH=/platform_tech/jyren/transpeeder/workspace/1013-polish/logs/1107-polish-slurm_log_$(date '+%m%d%H%M').txt
GPUS_PER_NODE=8
# 节点数
NNODES=4
# 总gpu数
N_GPUS=32

# testing for potential faulty nodes
# srun --jobid $SLURM_JOB_ID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
# exit 0

# 模型保存路径
OUTPUT=/shared_space/agpt/polish_sft/1107/output/${name}
if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

echo "conda env: $CONDA_PREFIX"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "

# 训练任务
export CMD=" \
    ${WORK_DIR}/examples/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt /shared_space/renjingyi/agpt/models/llama2-intros-init-ckpt \
    --data_path /shared_space/agpt/polish_sft/1107/data/train_1107_4096.jsonl \
    --max_seq_len 4096 \
    --train_steps 500 \
    --eval_steps 10 \
    --save_steps 50 \
    --log_steps 1 \
    --mode dialog \
    --pipe_parallel_size 16 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --ntk false \
    --deepspeed_config /platform_tech/jyren/transpeeder/workspace/1013-polish/1107-polish/ds_config_zero1_dgx_70B_8192.json
    "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

echo "START TIME: $(date)"

bash -c "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
