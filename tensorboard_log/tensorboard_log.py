from tensorboard.backend.event_processing import event_accumulator
import json
import os
import time
strtime = time.strftime("%Y%m%d")

# 指定你的log目录
log_dir = '/workspace/tensorboard_log/verl_grpo_wingpt/wingpt_30b/'
output_dir = '/workspace/LLM-Train/LLM-RL/LLM-veRL/tensorboard_log/'+strtime
os.makedirs(output_dir, exist_ok=True)

# 获取指定目录下的所有文件，并找到时间最新的events文件
def get_latest_event_file(log_dir):
    files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    return os.path.join(log_dir, latest_file)

latest_log_file = get_latest_event_file(log_dir)


# 创建EventAccumulator对象
ea = event_accumulator.EventAccumulator(latest_log_file,
                                         size_guidance={event_accumulator.SCALARS: 0})

# 加载事件数据
ea.Reload()

# 假设我们感兴趣的标签
interested_tags = ['actor/kl_loss', 'actor/entropy_loss', 'actor/grad_norm', 'critic/rewards/mean', 'response_length/mean', 'timing_s/step', 'critic/advantages/mean', 'val/test_score/unknown']

# 准备一个字典来存储数据
data_to_save = {}

# 提取每个感兴趣标签的数据
for tag in interested_tags:
    if tag in ea.Tags()['scalars']:
        # 获取指定tag下的所有数据
        events = ea.Scalars(tag)
        
        # 将数据转换成适合json的格式
        data_to_save[tag] = [{'step': e.step, 'value': e.value} for e in events]

# 定义输出json文件路径
output_json_path = os.path.join(output_dir, 'training_log_data.json')

# 写入json文件
with open(output_json_path, 'w') as f:
    json.dump(data_to_save, f, indent=4)

print(f"数据已成功写入{output_json_path}")