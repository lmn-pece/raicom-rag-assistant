import os

current_file = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _has_model_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    return "config.json" in names and (
        "model.safetensors" in names
        or "pytorch_model.bin" in names
        or "model.safetensors.index.json" in names
    )


def resolve_embed_model_dir() -> str:
    candidates = [
        os.path.join(MODELS_DIR, "baai", "bge-small-zh-v1___5"),
        os.path.join(MODELS_DIR, "baai", "bge-small-zh-v1.5"),
        os.path.join(MODELS_DIR, "AI-ModelScope", "bge-small-zh-v1-5"),
        os.path.join(MODELS_DIR, "AI-ModelScope", "bge-small-zh-v1___5"),
    ]
    for c in candidates:
        if _has_model_files(c):
            return c
    if os.path.isdir(MODELS_DIR):
        for base, dirs, files in os.walk(MODELS_DIR):
            if "config.json" in files and "bge" in base.lower():
                if _has_model_files(base):
                    return base
    return candidates[0]


EMBED_MODEL_DIR = os.environ.get("RAG_EMBED_MODEL_DIR") or resolve_embed_model_dir()
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
