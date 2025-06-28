# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Reranking dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
SYS_MSG = "You are CodeRanker, an intelligent code reviewer that can analyze GitHub issues and rank code functions based on their relevance to contain the faults causing the GitHub issue."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', required=True)
    parser.add_argument('--test_data_dir', required=True)
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # Load training dataset
    if ".json" in args.train_data_dir or ".jsonl" in args.train_data_dir:
        train_dataset = datasets.load_dataset("json", data_files=args.train_data_dir, split="train")
    else:
        train_dataset = datasets.load_dataset(args.train_data_dir, name="main", split="train")

    # Load test dataset
    if ".json" in args.test_data_dir or ".jsonl" in args.test_data_dir:
        test_dataset = datasets.load_dataset("json", data_files=args.test_data_dir, split="train")
    else:
        test_dataset = datasets.load_dataset(args.test_data_dir, name="main", split="test")

    # add a ow to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')

            answer = example.pop('solution')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "system",
                    "content": SYS_MSG,
                },{
                    "role": "user",
                    "content": question,
                }],
                "ability": "reranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', args.train_data_dir), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', args.test_data_dir), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)