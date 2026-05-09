import torch

# 模型配置
MODEL_NAME = "qwen/Qwen2.5-7B-Instruct"

# 4-bit 量化配置
QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

AGENT_SYSTEM_PROMPT = """你是一个睿康大赛机器人指挥官。
你可以调用以下工具来完成任务：
1. get_current_time: 获取当前时间。

当你需要调用工具时，请严格遵守以下格式输出：
Thought: 说明你为什么要调用这个工具
Action: 工具名称
Action Input: 工具参数内容

当你获得工具结果或直接回答时，输出：
Final Answer: 你的最终回复内容"""