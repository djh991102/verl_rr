#!/bin/bash

TEST_FILE="/mnt/nas/jaehyeok/cornstack/datasets/preprocessed/test.parquet"
Qwen3="Qwen/Qwen3-8B"
Qwen3_GRPO="/mnt/nas/jaehyeok/models/reranker/GRPO-SweRank-Qwen_step60"

MODEL_PATH=${Qwen3}

RESPONSE_LEN=4096

python scripts/model_evaluation.py \
    --parquet_path ${TEST_FILE} \
    --model_path ${MODEL_PATH} \
    --response_length ${RESPONSE_LEN}