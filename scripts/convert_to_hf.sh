#!/bin/bash

PROJECT_NAME="GRPO-SweRank"
EXP_NAME="GRPO-SweRank-Qwen"
MODEL_NAME="Qwen3-8B"
step=60

MODEL_PATH="/mnt/nas/jaehyeok/cornstack/ckpts/${PROJECT_NAME}/${EXP_NAME}/${MODEL_NAME}/global_step_${step}/actor"
SAVE_PATH="/mnt/nas/jaehyeok/models/reranker/${EXP_NAME}_step${step}"

# python scripts/model_merger.py \
#     --local_dir ${MODEL_PATH}


# If you want to move the model to a specific path, uncomment the following line
# mkdir -p ${SAVE_PATH}
# cp -r "${MODEL_PATH}/huggingface/"* ${SAVE_PATH}