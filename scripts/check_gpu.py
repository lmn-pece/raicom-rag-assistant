import torch
from bitsandbytes.nn import Linear4bit

# 尝试在显卡上创建一个 4-bit 的神经网络层
try:
    layer = Linear4bit(10, 10).cuda()
    print("✅ 显卡 4-bit 算力测试成功！你可以放心跑 Qwen 了。")
except Exception as e:
    print(f"❌ 显卡握手失败: {e}")