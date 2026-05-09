import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from modelscope import snapshot_download
from src.config import MODEL_NAME, QUANT_CONFIG


def load_qwen():
    print(f"正在准备模型资源: {MODEL_NAME}...")
    model_dir = snapshot_download(MODEL_NAME)

    q_config = BitsAndBytesConfig(**QUANT_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=q_config,
        trust_remote_code=True
    )
    return model, tokenizer