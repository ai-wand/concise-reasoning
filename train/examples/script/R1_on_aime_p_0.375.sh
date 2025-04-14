
HDFS_HOME=.
RUN_NAME=R1_on_aime_p_0.375

python3 openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 2 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 4 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 4 \
    --max_epochs 1 \
    --num_episodes 2000 \
    --prompt_max_len 1024 \
    --generate_max_len 20000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  data/aime_p_0.375_data_processed_with_r1_prompt.json \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 2000 \
    --load_checkpoint \
    --use_wandb YOUR_WANDB_KEY \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000 \
    --gamma 1 \
    --apply_chat_template \


