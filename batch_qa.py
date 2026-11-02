"""兼容旧入口：批量文档问答。

推荐在项目根目录使用:
    python -m scripts.batch_qa
或:
    python scripts/batch_qa.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.batch_qa import main

if __name__ == "__main__":
    main()