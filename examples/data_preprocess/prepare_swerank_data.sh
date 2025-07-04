DATA_PATH="/mnt/nas/jaehyeok/cornstack/datasets"
TRAIN_FILE="${DATA_PATH}/reranking_function_localization_grpo_train-train_sample.jsonl"
TEST_FILE="${DATA_PATH}/reranking_function_localization_grpo_train-test_sample.jsonl"
SAVE_DIR="${DATA_PATH}/preprocessed"

mkdir -p $SAVE_DIR

python examples/data_preprocess/reranking.py \
    --train_data_dir $TRAIN_FILE \
    --test_data_dir $TEST_FILE \
    --local_dir $SAVE_DIR