import json
import matplotlib.pyplot as plt
import os
import time
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.ticker as ticker

# 获取当前时间
strtime = time.strftime("%Y%m%d")

# 指定你的log目录
log_dir = '/root/tensorboard_log/verl_grpo_wingpt/wingpt_30b/'

# 获取指定目录下的所有文件，并找到时间最新的events文件
def get_latest_event_file(log_dir):
    files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not files:
        raise FileNotFoundError("No event files found in the specified directory.")
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    return os.path.join(log_dir, latest_file)

latest_log_file = get_latest_event_file(log_dir)

# 创建EventAccumulator对象
ea = event_accumulator.EventAccumulator(latest_log_file,
                                         size_guidance={event_accumulator.SCALARS: 0})

# 加载事件数据
ea.Reload()

# 指定感兴趣的标签
interested_tags = ['actor/kl_loss', 'actor/entropy_loss',  'actor/grad_norm', 
                   'critic/rewards/mean', 'response_length/mean', 'timing_s/step', 
                   'critic/advantages/mean', 'val-aux/unknown/reward/mean@1']

# 确保输出目录存在
output_dir = f'/workspace/LLM-Train/LLM-RL/LLM-veRL/tensorboard_log/{strtime}'  # 指定你想要保存图片的目录
os.makedirs(output_dir, exist_ok=True)

# 遍历每个感兴趣的标签，并为其绘制并保存图表
for tag in interested_tags:
    if tag in ea.Tags()['scalars']:
        # 提取步数和值
        steps = [item.step for item in ea.Scalars(tag)]
        values = [item.value for item in ea.Scalars(tag)]

        # 创建一个新的图形对象
        plt.figure(figsize=(10, 6))

        # 绘制数据
        plt.plot(steps, values, label=tag, color='b')

        # 添加标题和标签
        plt.title(f'Training Metric: {tag}')
        plt.xlabel('Step')
        plt.ylabel('Value')
        
        # 设置 x 轴刻度为整数格式
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # 显示图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # 保存图表到文件，文件名即为标签名
        output_image_path = os.path.join(output_dir, f'{tag.replace("/", "_")}.png')  # 使用replace来处理路径中的斜杠
        plt.savefig(output_image_path)

        # 关闭绘图以释放内存
        plt.close()

        print(f"图表已保存到 {output_image_path}")
    else:
        print(f"Tag '{tag}' not found in the event file.")

# # 遍历每个感兴趣的标签，并为其绘制并保存图表
# for tag in interested_tags:
#     if tag in ea.Tags()['scalars']:
#         # 提取步数和值
#         steps = [item.step for item in ea.Scalars(tag)]
#         values = [item.value for item in ea.Scalars(tag)]

#         # 创建一个新的图形对象
#         plt.figure(figsize=(10, 6))

#         # 绘制数据
#         plt.plot(steps, values, label=tag, color='b', marker='o')

#         # 在每个点上显示 y 值
#         for step, value in zip(steps, values):
#             plt.text(step, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

#         # 添加标题和标签
#         plt.title(f'Training Metric: {tag}')
#         plt.xlabel('Step')
#         plt.ylabel('Value')

#         # 设置 x 轴刻度为整数格式
#         plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

#         # 显示图例
#         plt.legend()

#         # 显示网格
#         plt.grid(True)

#         # 保存图表到文件，文件名即为标签名
#         output_image_path = os.path.join(output_dir, f'{tag.replace("/", "_")}.png')  # 使用replace来处理路径中的斜杠
#         plt.savefig(output_image_path)

#         # 关闭绘图以释放内存
#         plt.close()

#         print(f"图表已保存到 {output_image_path}")
#     else:
#         print(f"Tag '{tag}' not found in the event file.")
        
        
