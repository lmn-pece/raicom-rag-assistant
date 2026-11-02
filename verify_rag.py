"""兼容旧入口：检索与模型读取验证。

推荐在项目根目录使用:
    python -m scripts.verify_rag
或:
    python scripts/verify_rag.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.verify_rag import main

if __name__ == "__main__":
    main()