import json
from collections import defaultdict
import time
strtime = time.strftime("%Y%m%d")

# 基础参数
epoch_steps = 76
length_traindata = epoch_steps * 128
num_gen = 8


# 数据加载
input_dir = '/workspace/AAA-LLM-RL/LLM-veRL/log_date/20250513-log_train.json'
with open(input_dir, 'r', encoding='utf-8') as f:
    lists = [json.loads(line) for line in f]
train_result = [item for item in lists if len(item['reward_score']) > 1]

tensorboard_log_path = f'/workspace/AAA-LLM-RL/LLM-veRL/tensorboard_log/{strtime}/training_log_data.json'
tensorboard_log_data = json.load(open(tensorboard_log_path, 'r', encoding='utf-8'))

response_length = [k['value'] for k in tensorboard_log_data['response_length/mean']]

# 统计整体训练结果
def calculate_statistics(data):
    N = sum(1 for item in data if max(item['reward_score']) > 0)
    S = sum(1 for item in data if min(item['reward_score']) > 0)
    W = sum(1 for item in data if max(item['reward_score']) < 0)
    Y = sum(1 for item in data if '<|im_end|>' not in item.get('completion', ''))
    
    accu = round(N / len(data), 2) * 100 if data else 0
    return {
        'total': len(data),
        'correct_any': N,
        'correct_all': S,
        'incorrect_all': W,
        'truncated_all': Y,
        'accuracy': accu
    }

print("整体训练结果的统计")
overall_stats = calculate_statistics(train_result)
for key, value in overall_stats.items():
    print(f"{key}: {value}")
print("*********************************************\n")

# 按 epoch 分组
def group_by_epoch(data, length_per_epoch):
    return [data[i:i + length_per_epoch] for i in range(0, len(data), length_per_epoch)]

epochs = group_by_epoch(train_result, length_traindata)
response_lengths = group_by_epoch(response_length, epoch_steps)

# epoch 训练间对比统计
def compare_epochs(epoch1, epoch2):
    question_count = defaultdict(dict)
    for item in epoch1:
        if 'question' in item:
            question_count[item['question']]['ep1'] = item['reward_score']
    
    duplicates = []
    for item in epoch2:
        if item['question'] in question_count:
            ep1_scores = question_count[item['question']]['ep1']
            ep2_scores = item['reward_score']
            
            duplicates.append({
                'question': item['question'],
                'reward_score_ep1': ep1_scores,
                'reward_score_ep2': ep2_scores,
                'sum_ep1': sum(ep1_scores),
                'sum_ep2': sum(ep2_scores)
            })
    
    # 统计提升和下降的数量
    improved = sum(1 for d in duplicates if d['sum_ep2'] > d['sum_ep1'])
    declined = sum(1 for d in duplicates if d['sum_ep2'] < d['sum_ep1'])
    
    # 统计正确答案、全部正确、全部截断等
    correct_ep1 = sum(1 for d in duplicates if max(d['reward_score_ep1']) > 0)
    correct_ep2 = sum(1 for d in duplicates if max(d['reward_score_ep2']) > 0)
    all_correct_ep1 = sum(1 for d in duplicates if min(d['reward_score_ep1']) > 0)
    all_correct_ep2 = sum(1 for d in duplicates if min(d['reward_score_ep2']) > 0)
    all_incorrect_ep1 = sum(1 for d in duplicates if max(d['reward_score_ep1']) == -1)
    all_incorrect_ep2 = sum(1 for d in duplicates if max(d['reward_score_ep2']) == -1)
    
    format_violation_ep1 = sum(len([n for n in d['reward_score_ep1'] if n == -1]) for d in duplicates) / (len(duplicates) * num_gen)
    format_violation_ep2 = sum(len([n for n in d['reward_score_ep2'] if n == -1]) for d in duplicates) / (len(duplicates) * num_gen)
    
    return {
        'duplicates': duplicates,
        'improved': improved,
        'declined': declined,
        'correct_ep1': correct_ep1,
        'correct_ep2': correct_ep2,
        'all_correct_ep1': all_correct_ep1,
        'all_correct_ep2': all_correct_ep2,
        'all_incorrect_ep1': all_incorrect_ep1,
        'all_incorrect_ep2': all_incorrect_ep2,
        'format_violation_ep1': format_violation_ep1,
        'format_violation_ep2': format_violation_ep2
    }

# 多个 epoch 的对比
if len(epochs) >= 2:
    print("epoch 训练间对比统计")
    for i in range(len(epochs) - 1):
        epoch1 = epochs[i]
        epoch2 = epochs[i + 1]
        comparison_stats = compare_epochs(epoch1, epoch2)
        
        # 提取 sum_ep1 和 sum_ep2 并计算平均值
        sum_ep1_total = sum(d['sum_ep1'] for d in comparison_stats['duplicates'])
        sum_ep2_total = sum(d['sum_ep2'] for d in comparison_stats['duplicates'])
        avg_ep1 = sum_ep1_total / len(comparison_stats['duplicates']) if comparison_stats['duplicates'] else 0
        avg_ep2 = sum_ep2_total / len(comparison_stats['duplicates']) if comparison_stats['duplicates'] else 0
        
        print(f"\n对比 epoch{i+1} 和 epoch{i+2}:")
        print(f"已重复训练数据总量: {len(comparison_stats['duplicates'])}")
        print(f"已重复训练steps: {int(len(comparison_stats['duplicates'])/128)}")
        print(f"epoch{i+1} 训练时平均长度为:{round(sum(response_lengths[i]) / len(response_lengths[i]), 2)}")
        print(f"epoch{i+2} 训练时平均长度为:{round(sum(response_lengths[i+1]) / len(response_lengths[i+1]), 2)}")
        print(f"epoch{i+1} 平均分数: {avg_ep1}")
        print(f"epoch{i+2} 平均分数: {avg_ep2}")
        print(f"与前一轮训练时结果提升的数据数量: {comparison_stats['improved']}")
        print(f"与前一轮训练时结果下降的数据数量: {comparison_stats['declined']}")
        print(f"epoch{i+1} 训练时含有正确答案为: {comparison_stats['correct_ep1'] / len(comparison_stats['duplicates']) if comparison_stats['duplicates'] else 0}")
        print(f"epoch{i+2} 训练时含有正确答案为: {comparison_stats['correct_ep2'] / len(comparison_stats['duplicates']) if comparison_stats['duplicates'] else 0}")
        print(f"epoch{i+1} 样本答案全部正确的数据量: {comparison_stats['all_correct_ep1']}")
        print(f"epoch{i+2} 样本答案全部正确的数据量: {comparison_stats['all_correct_ep2']}")
        print(f"epoch{i+1} 训练时样本答案全部截断数据量: {comparison_stats['all_incorrect_ep1']}")
        print(f"epoch{i+2} 训练时样本答案全部截断数据量: {comparison_stats['all_incorrect_ep2']}")
        print(f"epoch{i+1} 格式不遵循样本比例: {comparison_stats['format_violation_ep1']}")
        print(f"epoch{i+2} 格式不遵循样本比例: {comparison_stats['format_violation_ep2']}")
else:
    print("数据不足两个 epoch，无法进行对比统计。\n")
