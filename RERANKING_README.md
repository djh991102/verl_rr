## Environment setup
```
conda create -n verl python==3.10
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

# If vLLM is not installed
pip install vllm==0.8.3
pip install flash-attn --no-build-isolation
pip install tensordict==0.6.2
```

### Features

vLLM 0.8+ supports cuda graph and V1 engine by default in veRL. To enable these features, remember to add the following lines to the bash script:

```bash
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \
```

## Reranker Oneline RL
### Data Preparation

To prepare the data for reranker training, use the script `examples/data_preprocess/prepare_swerank_data.sh`. This script processes the jsonl reranking data into parquet for GRPO training.

The script takes the inputs:
- `TRAIN_FILE`: Path to training data in JSONL format
- `TEST_FILE`: Path to test data in JSONL format  
- `SAVE_DIR`: Path to save data in parquet format

```bash
bash examples/data_preprocess/prepare_swerank_data.sh
```

### GRPO
```bash
bash examples/grpo_trainer/run_grpo_swerank.sh
```

### DAPO
```bash
bash recipe/dapo/run_dap_swerank.sh
```

