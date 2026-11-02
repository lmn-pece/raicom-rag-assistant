from modelscope import snapshot_download
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_save_path = os.path.join(BASE_DIR, "models")

print(" 尝试从 ModelScope 下载 BGE 模型...")

try:
    # 修正后的 ID：直接使用 'ZhipuAI/bge-small-zh-v1.5' 或 'baai/bge-small-zh-v1.5'
    # 在 ModelScope 上，这个模型最稳妥的 ID 通常是以下这个：
    model_dir = snapshot_download(
        'AI-ModelScope/bge-small-zh-v1-5',  # 如果这个报错，尝试换成 'baai/bge-small-zh-v1.5'
        cache_dir=model_save_path,
        revision='master'  # 强制指定分支，防止 404
    )
    print("-" * 30)
    print(f" 下载完成！路径: {os.path.abspath(model_dir)}")

except Exception as e:
    print(f" 下载失败: {e}")
    print(" 备选方案：正在尝试使用替代 ID...")
    # 备选 ID
    model_dir = snapshot_download(
        'baai/bge-small-zh-v1.5',
        cache_dir=model_save_path
    )
    print(f"备选 ID 下载完成！路径: {os.path.abspath(model_dir)}")