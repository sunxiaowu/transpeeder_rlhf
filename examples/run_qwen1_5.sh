export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 22395 train_qwen2.py \
    --output_dir /platform_tech/models/Qwen-1.5-4b-chat-ckpt2-sft \
    --init_ckpt /platform_tech/models/Qwen1.5-4b-chat-ckpt2 \
    --data_path /home/xuxiaolong/reading_copilot/1b_sft/fudu_openorca_clean.jsonl \
    --max_seq_len 12768 \
    --train_steps 10000 \
    --eval_steps 500 \
    --save_steps 500 \
    --log_steps 1 \
    --pipe_parallel_size 8  \
    --model_parallel_size 1 \
    --deepspeed_config /home/xuxiaolong/AcademicGPT/transpeeder/configs/ds_config_zero1_agent_qwen2_7b.json 1>log_qwen2-7b-mp1-108k-20240308.log 2>&1 &