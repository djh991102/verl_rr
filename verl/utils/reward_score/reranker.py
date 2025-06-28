import re

response_pattern = response_pattern = r"^\s*?<think>.*?</think>\s*?<answer>\s*?(\[\d+\](?:\s*>\s*\[\d+\])*)\s*?</answer>\s*?$"

def get_gold_rank(ranked_list: str, ground_truth: str):
    # With proper extraction, expect ranked_list to be of the form [] > ... > []
    doc_ids = [doc_id.strip() for doc_id in ranked_list.split('>')]
    if ground_truth in doc_ids:
        return doc_ids.index(ground_truth)
    else:
        return None

def topk_inverse_rank_score(data_source, solution_str, ground_truth, extra_info=None, k=5):
    solution_str = solution_str.strip()
    ground_truth = ground_truth.strip()
    match = re.findall(response_pattern, solution_str, re.DOTALL | re.MULTILINE)
    if match:
        ranked_list = match[-1]
        # compute inverse rank score if gold within topK else reward with minimal value
        # extract ranked output
        gold_rank = get_gold_rank(ranked_list, ground_truth)
        if gold_rank is not None and gold_rank < k:
            # reward with inverse rank score
            inv_rank_score = 1/(gold_rank+1)
            
            # minmax normalization to keep the score within a range [0,1]
            min_score=1/(k+1)
            max_score=1
            inv_rank_score = (inv_rank_score-min_score)/(max_score-min_score)
            return inv_rank_score
    # incorrect format or bad prediction (gold rank > topK) 
    return 0

def topk_presence_score(data_source, solution_str, ground_truth, extra_info=None, k=5):
    solution_str = solution_str.strip()
    ground_truth = ground_truth.strip()
    match = re.findall(response_pattern, solution_str, re.DOTALL | re.MULTILINE)
    if match:
        ranked_list = match[-1]
        gold_rank = get_gold_rank(ranked_list, ground_truth)

        # reward based on the gold's presence within topK 
        if gold_rank is not None and gold_rank < k:
            return 1
    # incorrect format or bad prediction (gold rank > topK)
    return 0