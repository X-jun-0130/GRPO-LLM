import json
from collections import defaultdict
import time
from verl import DataProto
import torch
from rewards import accuracy_reward

strtime = time.strftime("%Y%m%d")

def process_questions(data):
    # 按问题分组存储所有条目
    question_groups = defaultdict(list)
    for item in data:
        question_groups[item['question']].append(item)

    # 处理每个分组
    processed = []
    for question, items in question_groups.items():
        # 收集所有评分并找到最高分条目
        reward_scores = []
        max_score = float('-inf')
        best_item = None
        
        for item in items:
            current_score = item['reward_score']
            reward_scores.append(current_score)
            
            if current_score > max_score:
                max_score = current_score
                best_item = item

        # 构建新条目
        processed.append({
            'question': question,
            'completion': best_item['completion'],
            'solution': best_item['solution'],
            'reward_score': reward_scores  # 包含所有历史评分的列表
        })
    return processed


###  多线程的处理模式   
from concurrent.futures import ThreadPoolExecutor
class MultiRewardManager():
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def process_single_item(self, args):
        i, data_item = args
        messages = data_item.non_tensor_batch['messages']
        task = data_item.non_tensor_batch['data_task']
        limit = data_item.non_tensor_batch['limits']
        
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        #valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        output = response_str.strip()
        
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        
        length = valid_response_length
        score = accuracy_reward(messages, output, ground_truth, task, limit, length)
        
        return (i, score, output, messages, ground_truth, valid_response_length)

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        data_list = []
        
        # Separate math and non-math tasks
        math_args = []
        other_args = []
        
        for i in range(len(data)):
            data_item = data[i]
            task = data_item.non_tensor_batch['data_task']
            if task in ['math', 'MedCalc-Bench']:
                math_args.append((i, data_item))
            else:
                other_args.append((i, data_item))
        
        # Process math tasks sequentially
        math_results = []
        for args in math_args:
            math_results.append(self.process_single_item(args))
        
        # Process other tasks in parallel
        other_results = []
        if other_args:
            with ThreadPoolExecutor(max_workers=128) as executor:
                other_results = list(executor.map(self.process_single_item, other_args))
        
        # Combine all results
        all_results = math_results + other_results
        
        for res in all_results:
            i, score, output, messages, ground_truth, valid_response_length = res
            
            data_list.append({
                'question': messages[-1]['content'],
                'completion': output,
                'solution': ground_truth,
                'reward_score': score
            })
            

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            reward_tensor[i, valid_response_length - 1] = reward

        try:
            process_data = process_questions(data_list)
            with open(f'/workspace/LLM-Train/LLM-RL/LLM-veRL/log_date/{strtime}-log_train.json', 'a+', encoding='utf-8') as target_file:
                for item in process_data:
                    target_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        except:
            pass
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    

class DAPORewardManager():
    """The reward manager.
    """
    def __init__(self, 
                 tokenizer, 
                 num_examine,
                 max_resp_len=None,
                 overlong_buffer_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def process_single_item(self, args):
        i, data_item = args
        messages = data_item.non_tensor_batch['messages']
        task = data_item.non_tensor_batch['data_task']
        limit = data_item.non_tensor_batch['limits']
        
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        #valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        output = response_str.strip()
        
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        
        length = valid_response_length
        score = accuracy_reward(messages, output, ground_truth, task, limit, length)
        
        if self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            score += overlong_reward
        
        return (i, score, output, messages, ground_truth, valid_response_length)

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
            
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        data_list = []
        
        # Separate math and non-math tasks
        math_args = []
        other_args = []
        
        for i in range(len(data)):
            data_item = data[i]
            task = data_item.non_tensor_batch['data_task']
            if task in ['math', 'MedCalc-Bench']:
                math_args.append((i, data_item))
            else:
                other_args.append((i, data_item))
        
        # Process math tasks sequentially
        math_results = []
        for args in math_args:
            math_results.append(self.process_single_item(args))
        
        # Process other tasks in parallel
        other_results = []
        if other_args:
            with ThreadPoolExecutor(max_workers=128) as executor:
                other_results = list(executor.map(self.process_single_item, other_args))
        
        # Combine all results
        all_results = math_results + other_results
        
        for res in all_results:
            i, score, output, messages, ground_truth, valid_response_length = res
            
            data_list.append({
                'question': messages[-1]['content'],
                'completion': output,
                'solution': ground_truth,
                'reward_score': score
            })
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            reward_tensor[i, valid_response_length - 1] = reward

        try:
            process_data = process_questions(data_list)
            with open(f'/workspace/LLM-Train/LLM-RL/LLM-veRL/log_date/{strtime}-log_train.json', 'a+', encoding='utf-8') as target_file:
                for item in process_data:
                    target_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        except:
            pass

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    


