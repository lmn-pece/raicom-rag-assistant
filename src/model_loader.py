import os
from typing import Optional

os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import MODEL_NAME, QUANT_CONFIG


def _normalize_name(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _looks_like_causal_model(path: str) -> bool:
    names = set(os.listdir(path)) if os.path.isdir(path) else set()
    if "config.json" not in names:
        return False
    return bool(
        {"model.safetensors.index.json", "pytorch_model.bin.index.json"} & names
        or "model.safetensors" in names
        or "pytorch_model.bin" in names
    )


def _find_local_model(name: str) -> Optional[str]:
    cache_root = os.environ.get("MODELSCOPE_CACHE") or os.path.expanduser("~/.cache/modelscope")
    roots = [
        os.path.join(cache_root, "hub", "models"),
        os.path.join(cache_root, "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"),
    ]
    target = _normalize_name(name.split("/")[-1])
    exact_matches = []
    fuzzy_matches = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for base, _, _ in os.walk(root):
            if not _looks_like_causal_model(base):
                continue
            norm_base = _normalize_name(os.path.basename(base))
            if norm_base.endswith(target) or target in norm_base:
                exact_matches.append(base)
            elif "qwen" in base.lower() and target in _normalize_name(base.replace(os.sep, "/")):
                fuzzy_matches.append(base)
    if exact_matches:
        return sorted(exact_matches, key=len)[-1]
    if fuzzy_matches:
        return sorted(fuzzy_matches, key=len)[-1]
    return None


def resolve_model_dir() -> str:
    env_path = os.environ.get("RAG_LLM_MODEL_DIR")
    if env_path and os.path.isdir(env_path):
        return env_path
    if os.path.isdir(MODEL_NAME):
        return MODEL_NAME
    local = _find_local_model(MODEL_NAME)
    if local:
        return local
    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            f"未找到本地大模型，且 modelscope 不可用。请设置 RAG_LLM_MODEL_DIR。目标: {MODEL_NAME}"
        ) from exc
    try:
        return snapshot_download(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"无法加载本地大模型 {MODEL_NAME}，请先下载或设置 RAG_LLM_MODEL_DIR。"
        ) from exc


def load_qwen():
    model_dir = resolve_model_dir()
    print(f"正在加载本地模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    use_cuda = torch.cuda.is_available()
    has_bnb = False
    if use_cuda:
        try:
            import bitsandbytes  # noqa: F401

            has_bnb = True
        except Exception:
            has_bnb = False

    if use_cuda and has_bnb:
        kwargs["quantization_config"] = BitsAndBytesConfig(**QUANT_CONFIG)
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    elif use_cuda:
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return model, tokenizer
