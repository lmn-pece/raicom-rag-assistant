# src/core/agent_logic.py
import datetime

def get_current_time(*args, **kwargs):
    """获取当前系统时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 工具映射表
TOOLS = {
    "get_current_time": get_current_time,
}