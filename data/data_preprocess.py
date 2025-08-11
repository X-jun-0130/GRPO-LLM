"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""
import os
from datasets import load_dataset

'''
{'id':"",'messages':"", 'solution':"", 'task':"", 'limits':""}
messages表示输入问题，单轮或多轮对话的形式
solution表示已审核的答案
task表示任务名称，不同任务采用不同的奖励方法
limits表示区间范围，有些任务答案在范围内都算正确
'''

# Load the dataset
dataset = load_dataset("json", data_files='/workspace/LLM-Train/LLM-RL/LLM-veRL/data/20250809.jsonl', split="train", cache_dir="/workspace/cache_dir/")
testdata_set = load_dataset("json", data_files='/workspace/LLM-Train/LLM-RL/LLM-veRL/data/Model_Test.jsonl', split="train", cache_dir="/workspace/cache_dir/")

# 打乱数据集，注意设置一个种子值以确保结果可重复
shuffled_traindataset = dataset.shuffle(seed=42)
shuffled_testdataset = testdata_set.shuffle(seed=42)
  
def make_map_fn(split):
    def process_fn(example, idx):
        data = {
            "data_task": example["task"],
            "prompt": example["messages"],
            "reward_model": {
                "style": "rule",
                "ground_truth": example["solution"]
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    return process_fn

local_dir = '/workspace/LLM-Train/LLM-RL/LLM-veRL/Model_Train_Data'

train_dataset = shuffled_traindataset.map(function=make_map_fn('train'), with_indices=True)
train_dataset.to_parquet(os.path.join(local_dir, 'train_grpo.parquet'))
print(len(train_dataset))
print(train_dataset[0])

test_dataset = shuffled_testdataset.map(function=make_map_fn('test'), with_indices=True)
test_dataset.to_parquet(os.path.join(local_dir, 'test_grpo.parquet'))
print(len(test_dataset))
print(test_dataset[0])
