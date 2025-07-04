import torch
import pyarrow.parquet as pq
from verl.utils.reward_score.reranker import topk_inverse_rank_score
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    # Load the pyarrow dataset
    table = pq.read_table(args.parquet_path)
    dataset = table.to_pandas()

    # Load the vllm model
    llm = LLM(model=args.model_path, dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(max_tokens=args.response_length, temperature=0.6)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    conversations = dataset['prompt']
    ground_truths = [x['answer'] for x in dataset['extra_info']]

    input_texts = [tokenizer.apply_chat_template(conversation, tokenize=False, enable_thinking=True, add_generation_prompt=True) for conversation in conversations]

    outputs = llm.generate(input_texts, sampling_params)
    preds = [output.outputs[0].text for output in outputs]

    # Evaluate with the custom reward function
    total_score = 0
    for input_text, pred, ground_truth in zip(input_texts, preds, ground_truths):
        score = topk_inverse_rank_score(None, pred, ground_truth, k=5)
        total_score += score

    avg_score = total_score / len(preds)
    print(f"Average Reward Score: {avg_score}")

if __name__ == "__main__":  
    args = argparse.ArgumentParser()
    args.add_argument("--parquet_path", type=str, default="test.parquet")
    args.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
    args.add_argument("--response_length", type=int, default=4096)
    args = args.parse_args()

    main(args)