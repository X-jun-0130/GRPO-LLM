"""Reward functions for GRPO training."""
import re
from math_verify import parse, verify
import time
from reward_model_api import text_generate
import json

strtime = time.strftime("%Y%m%d")


verifier_prompt:str = '''
你是一名专业的评估专家，需根据以下四个核心要素来对「预测答案」进行质量评估：

- **对话历史**（上下文信息）
- **当前问题**（用户提出的具体请求）
- **优秀答案**（经过审核的高质量参考答案）
- **预测答案**（待评估的答案）

### ⭐ 评分标准：
- **满分**：满分100分，表示「预测答案」质量高，与「优秀答案」相当或接近，满足用户需求；
- **扣分**：根据「预测答案」存在的问题---幻觉、遗漏、错误或无法满足用户需求等，进行相应的扣分；

> 注意：「优秀答案」已通过严格审核，其质量被认为是高标准的，可作为判断基准。

请按照以下结构输出你的评估结果：

---

### 📜 对话历史（按时间顺序排列，从最早到最新）
```
{插入对话历史}
```

### ❓ 当前问题
```
{插入原始问题}
```

### ✅ 优秀答案（参考答案）
```
assistant：{插入优秀答案}
```

### 🤖 预测答案（待评估答案）
```
assistant：{插入待评估答案}
```

---

### 📊 评估分析

[在此处进行逐项对比分析]

---

### 📌 预测答案评估分数

\\boxed{预测答案评估分数}

---
'''


def extract_evaluation_score(response: str) -> float:
    """
    从模型回复中提取评估分数（最后一个\\boxed{}中的0-100分数字），
    处理整数和小数格式，并将结果压缩到0-1范围
    
    参数:
        response: 包含评估分数的文本字符串
        
    返回:
        提取并处理后的分数（0.0-1.0之间的浮点数）
    """
    # 匹配所有\boxed{}内容
    boxes = re.findall(r'\\boxed\{([^}]*)\}', response)
    
    if not boxes:
        return 0.0
    
    # 取最后一个boxed内容
    last_box = boxes[-1].strip()
    
    # 改进的数字匹配：支持整数和小数格式（如95, 96.5, 85分）
    score_match = re.search(
        r'(\d{1,3}(?:\.\d{1,2})?)\s*(?:分)?\b', 
        last_box
    )
    
    if score_match:
        try:
            score = float(score_match.group(1))
            # 处理分数范围
            if score <= 40.0:  # 低分直接归零
                return 0.0
            elif score > 100.0:  # 超百分制处理
                return min(score / 100.0, 1.0)
            else:
                return score / 100.0
        except (ValueError, TypeError):
            return 0.0
    
    # 额外尝试匹配带等号的分数（如"=96.5"）
    equals_match = re.search(r'=\s*(\d{1,3}(?:\.\d{1,2})?)\b', last_box)
    if equals_match:
        try:
            score = float(equals_match.group(1))
            return max(0.0, min(score / 100.0, 1.0)) if score > 40.0 else 0.0
        except (ValueError, TypeError):
            pass
    
    return 0.0


def think_format_reward(content):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*\n</think>\n.+"
    flags = re.DOTALL  # 允许 . 匹配换行符
    required_tags = ['<think>', '</think>']
    
    # 检查整体结构是否符合正则表达式
    struct_match = re.fullmatch(pattern, content, flags=flags)
    if not struct_match:
        return False

    # 检查每个标签是否恰好出现一次
    tag_counts = [content.count(tag) for tag in required_tags]
    if all(count == 1 for count in tag_counts):
        return True
    else:
        return False


def get_reward(message, output, target, task, limit):
    if task == "quality-control":
        matches_text1 = re.findall(r'(true|false)', output)
        matches_text2 = re.findall(r'(true|false)', target)
        if matches_text1 == matches_text2:
            reward = 1.0
        else:
            reward = 0.0
                   
    elif task == 'math':
        answer = parse(output)
        reward = float(verify(answer, parse(target)))
        
    elif task == 'choice':
        matches_text1 = re.findall(r'\\boxed\{(?:\\text\{)?([ABCDEFG])(?:\..*)?(?:\})?\}', output)
        matches_text2 = re.findall(r'\\boxed\{(?:\\text\{)?([ABCDEFG])(?:\..*)?(?:\})?\}', target)
        if matches_text1 == matches_text2:
            reward = 1.0
        else:
            reward = 0.0
    
    elif task == 'MedCalc-Bench':
        answer = parse(output)[0]
        if limit[0] <= answer <= limit[1]:
            reward = 1.0
        else:
            reward = 0.0
             
    else:
        prompt = verifier_prompt.strip()

        if len(message)>=3:
            chathistory = message[:-1]
            history  = '\n'.join([k['role']+'：'+k['content'].strip() for k in chathistory])
            prompt = prompt.replace('{插入对话历史}', history)
            question = 'user：' + message[-1]['content'].strip()
        else:
            prompt = prompt.replace('{插入对话历史}', '无')
            question = '\n'.join([k['role']+'：'+k['content'].strip() for k in message])

        prompt = prompt.replace('{插入原始问题}', question.strip())
        prompt = prompt.replace('{插入优秀答案}', target.strip())
        prompt = prompt.replace('{插入待评估答案}', output.strip())
        
        # 模型处理流程
        chat = [{"role": "user", "content": prompt}]
        verifier_answer = text_generate(chat)

        reward = extract_evaluation_score(verifier_answer.strip())
            
        try:
            with open(f'/workspace/LLM-Train/LLM-RL/LLM-veRL/log_date/{strtime}-model_verifier.json', 'a+', encoding='utf-8') as target_file:
                target_file.write(json.dumps({'question':question,'solution':target, 'output':output, 'generated':verifier_answer, 'reward_score':reward}, ensure_ascii=False) + '\n')
        except:
            pass
    
    return reward


def calculate_reward(output_length):
    soft, hard, max_len = 4096, 8192, 24*1024
    min_reward = 0.1
    
    if output_length <= soft:
        return 1.0
    
    #温和线性惩罚 从1.0线性降至0.9
    if output_length <= hard:
        ratio = (output_length - soft) / (hard - soft)
        return round(1.0 - 0.1 * ratio, 4)
    
    #加速非线性惩罚 二次方加速下降曲线：0.9 -> 0.1
    ratio = (output_length - hard) / (max_len - hard)
    reward = 0.9 - 0.8 * (ratio ** 2)
    return round(max(reward, min_reward), 4)


def accuracy_reward(messages, content, solution, task, limit, output_length):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    # Shorter correct solutions are rewarded more than longer ones. add cosine
    
    try:
        if not content.endswith('<|im_end|>'):
            return 0.0
        else:
            content = content.replace('<|im_end|>', '')
            output = content.strip()
            base_reward = get_reward(messages, output, solution, task, limit)
            return base_reward
          
    except Exception:  # if it fails for any reason, return 0.0
        return 0.0
