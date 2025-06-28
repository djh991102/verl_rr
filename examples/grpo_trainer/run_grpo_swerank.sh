set -x

# Experiment
model_name='Qwen/Qwen3-8B'
project_name='GRPO-SweRank'
exp_name="GRPO-SweRank-${model_name}"

# Paths
REPO_PATH="/mnt/nas/jaehyeok/cornstack"
TRAIN_FILE="${REPO_PATH}/datasets/preprocessed/train.parquet"
TEST_FILE="${REPO_PATH}/datasets/preprocessed/test.parquet"

MODEL_PATH="${model_name}"
CKPTS_DIR="${REPO_PATH}/ckpts/${project_name}/${exp_name}"

# Training Algorithm
ALGORITHM_ARGS="algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001"

# Data
MAX_PROMPT_LEN=20000
MAX_COMPLETION_LEN=1024
DATA_ARGS="data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=1024 \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_COMPLETION_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error"
    
### ACTOR
# Actor training arguments
ACTOR_ARGS="actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(($MAX_PROMPT_LEN+$MAX_COMPLETION_LEN)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.optim.lr=1e-6"

# Actor model arguments
ACTOR_MODEL_ARGS="actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True"

# Actor rollout(generation) arguments
ACTOR_ROLLOUT_ARGS="actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2*($MAX_PROMPT_LEN+$MAX_COMPLETION_LEN))) \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2*($MAX_PROMPT_LEN+$MAX_COMPLETION_LEN)))" # For rollout tuning

# Reference model arguments
REF_ARGS="actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2*($MAX_PROMPT_LEN+$MAX_COMPLETION_LEN))) \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2
"

### GRPO REPLACES CRITIC WITH GROUP COMPUTATION (refer to grpo advantage computation)

# GRPO trainer arguments
TRAINER_ARGS="trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=${CKPTS_DIR}"

# Custom reward file path and function name
REWARD_ARGS="custom_reward_function.path=verl/utils/reward_score/reranker.py \
    custom_reward_function.name=topk_inverse_rank_score"


python -m verl.trainer.main_ppo \
    ${ALGORITHM_ARGS} \
    ${DATA_ARGS} \
    ${ACTOR_ARGS} \
    ${ACTOR_MODEL_ARGS} \
    ${REF_ARGS} \
    ${ACTOR_ROLLOUT_ARGS} \
    ${REWARD_ARGS} \
    ${TRAINER_ARGS} $@