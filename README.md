# RAICOM RAG Assistant

面向睿抗机器人开发者大赛（RAICOM）样题的本地检索增强问答（RAG）项目。系统以本地文档为第一知识来源，使用 Qwen2.5-7B-Instruct 生成回答，使用 BGE-small-zh-v1.5 做向量检索，并结合 BM25 完成混合召回。

## 核心特性

- **本地优先**：回答前先检索 `data/` 下的本地文档，禁止直接依赖模型训练记忆回答比赛题目。
- **题目级分块**：自动识别题号、题干、选项和题型，避免多道题被截断或混在一起。
- **混合检索**：向量相似度 + BM25 + 题号/题干精确匹配加权。
- **可追溯回答**：回答末尾附带来源文件、行号和 chunk_id。
- **无答案处理**：文档没有答案字段时，明确说明“本地资料中未提供答案”；AI 推断内容会标注为非本地资料。
- **结构化统计**：支持“共有多少道题”“列出所有题号”“把题目读出来”等问题。
- **可验证**：提供检索命中率和模型读取题目的自动化验证脚本。

## 项目结构

```text
AIProject/
├─ app.py                     # 交互式问答入口
├─ batch_qa.py                # 批量问答兼容入口
├─ verify_rag.py              # 验证脚本兼容入口
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ .github/
│  └─ workflows/tests.yml     # GitHub Actions 单元测试
├─ data/
│  └─ 样题.txt                # 本地知识库文档
├─ models/                    # 本地模型目录（不提交）
│  └─ baai/bge-small-zh-v1___5/
├─ storage/                   # RAG 索引缓存（不提交）
│  ├─ rag_chunks.json
│  ├─ rag_embeddings.pt
│  └─ rag_meta.json
├─ reports/
│  └─ verify_report.md        # 验证报告
├─ scripts/
│  ├─ batch_qa.py             # 批量问答实现
│  ├─ verify_rag.py           # 检索与模型读取验证实现
│  ├─ check_gpu.py            # 4-bit GPU 环境检查
│  └─ download_model.py       # BGE 模型下载辅助脚本
├─ src/
│  ├─ config.py               # 模型、检索和提示词配置
│  ├─ model_loader.py         # 本地大模型加载
│  ├─ core/
│  │  ├─ agent_logic.py       # 检索器单例与工具注册
│  │  ├─ brain.py             # 本地优先问答 Agent
│  │  ├─ question_bank.py     # 样题结构化解析
│  │  └─ rag_engine.py        # 分块、索引和混合检索
│  └─ utils/
│     └─ path.py              # 项目路径与模型路径解析
└─ tests/
   └─ test_docqa_extract.py   # 解析与题目结构测试
```

`models/`、`storage/`、`reports/verify_report.json` 不进入版本库。索引和报告都可以重新生成。

## 环境要求

- Python 3.10
- 支持 CUDA 的 NVIDIA GPU（推荐 8GB 及以上显存）
- 已安装 `requirements.txt` 中的依赖

本项目在 WSL2 + Ubuntu 22.04 + RTX 4060 Laptop（8GB）环境验证。Windows 下请使用已安装依赖的 Python 环境，或使用 WSL。

安装依赖：

```bash
pip install -r requirements.txt
```

也可以安装为可编辑包：

```bash
pip install -e .
```

如果使用 GPU，请确保安装的是对应 CUDA 版本的 PyTorch。`bitsandbytes` 用于 4-bit 量化加载 Qwen。

## 模型准备

### BGE 向量模型

默认按以下顺序查找本地 BGE 模型：

1. `models/baai/bge-small-zh-v1___5`
2. `models/baai/bge-small-zh-v1.5`
3. `models/AI-ModelScope/bge-small-zh-v1-5`
4. `models/` 下任意包含 `config.json` 且名称含 `bge` 的目录

也可以通过环境变量指定：

```bash
export RAG_EMBED_MODEL_DIR=/path/to/bge-small-zh-v1.5
```

辅助下载脚本：

```bash
python scripts/download_model.py
```

### Qwen 大模型

默认模型为 `qwen/Qwen2.5-7B-Instruct`。模型加载器会依次尝试：

1. 环境变量 `RAG_LLM_MODEL_DIR`
2. 项目内同名路径
3. 本地 ModelScope / HuggingFace 缓存
4. ModelScope 在线下载（离线环境会给出明确错误）

手动指定模型目录：

```bash
export RAG_LLM_MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
```

## 快速开始

### 1. 交互式问答

```bash
python app.py
```

支持命令：

- `exit` / `quit`：退出
- `clear`：清空对话记忆

### 2. 检索验证（不加载大模型）

```bash
python scripts/verify_rag.py
```

兼容旧入口：

```bash
python verify_rag.py
```

默认输出：

- `reports/verify_report.md`
- `reports/verify_report.json`

### 3. 检索 + 模型读取验证

```bash
python scripts/verify_rag.py --model --qids 1,51,71
```

只测试前若干题：

```bash
python scripts/verify_rag.py --model --limit 3
```

### 4. 批量文档问答

```bash
python scripts/batch_qa.py --questions data/样题.txt --max-questions 5
```

兼容旧入口：

```bash
python batch_qa.py --questions data/样题.txt --max-questions 5
```

默认输出：

- `answers.md`
- `answers.jsonl`

### 5. 运行测试

```bash
python -m unittest discover -s tests -v
```

也可以直接运行：

```bash
python tests/test_docqa_extract.py
```

### 6. GPU 环境检查

```bash
python scripts/check_gpu.py
```

该脚本会尝试在 GPU 上创建 4-bit 线性层，用于确认 `bitsandbytes` 和 CUDA 环境可用。

## 验证结果

最近一次验证结果见 `reports/verify_report.md`：

- 样题解析数量：72
- 检索完整命中：72/72
- 目标题目进入 Top-4：72/72
- 模型读取抽查第 1、51、71 题：3/3 通过

验证脚本会检查：

- 检索片段是否包含完整题干；
- 是否包含该题全部选项；
- 目标题号是否进入 Top-K；
- 模型输出是否完整复现题干和选项。

## 工作流程

1. `question_bank.py` 解析 `data/样题.txt`，提取题号、题干、选项、题型和行号。
2. `rag_engine.py` 按题目生成 chunk，并计算 BGE 向量与 BM25 索引。
3. 查询时同时进行向量检索、BM25 检索和题号/题干精确匹配加权。
4. `brain.py` 将检索到的本地资料注入提示词，要求模型只依据资料回答。
5. 回答附带来源文件和行号；无资料时拒答；无答案但需要推断时明确标注非本地资料。

## 配置说明

主要配置位于 `src/config.py`：

| 配置项 | 说明 |
| --- | --- |
| `MODEL_NAME` | 默认大模型名称 |
| `MAX_NEW_TOKENS` | 单次生成最大 token 数 |
| `RAG_TOP_K` | 默认检索片段数量 |
| `ALLOW_MODEL_INFERENCE` | 本地资料无答案时是否允许 AI 推断 |
| `INFERENCE_WARNING` | AI 推断的固定提示前缀 |

相关环境变量：

| 环境变量 | 说明 |
| --- | --- |
| `RAG_LLM_MODEL_DIR` | 指定 Qwen 模型目录 |
| `RAG_EMBED_MODEL_DIR` | 指定 BGE 模型目录 |
| `MODELSCOPE_OFFLINE` | 是否启用 ModelScope 离线模式 |
| `TRANSFORMERS_OFFLINE` | 是否启用 Transformers 离线模式 |

## 索引与缓存

索引文件位于 `storage/`，不会提交到 GitHub：

- `rag_chunks.json`：切分后的文本块及元数据
- `rag_embeddings.pt`：BGE 向量
- `rag_meta.json`：数据签名，用于判断是否需要重建

文档大小或修改时间变化时，索引会自动重建。也可以强制重建：

```bash
python scripts/verify_rag.py --rebuild-index
```

## 常见问题

### 1. 报错 `ModuleNotFoundError: No module named 'torch'`

当前 Python 环境没有安装项目依赖。请切换到已安装依赖的环境，或执行：

```bash
pip install -r requirements.txt
```

### 2. 找不到 BGE 模型

确认模型目录中存在 `config.json`，或设置：

```bash
export RAG_EMBED_MODEL_DIR=/path/to/bge-small-zh-v1.5
```

### 3. 找不到 Qwen 模型

确认 ModelScope 缓存目录，或设置：

```bash
export RAG_LLM_MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
```

### 4. 显存不足

- 关闭其他占用 GPU 的程序。
- 确认 `bitsandbytes` 已安装。
- 运行 `python scripts/check_gpu.py` 检查 4-bit GPU 环境。

### 5. 检索结果没有更新

删除 `storage/` 下的 `rag_*` 文件，或执行：

```bash
python scripts/verify_rag.py --rebuild-index
```

## 已知限制

- 样题文件只包含题干和选项，不包含标准答案；系统不会把 AI 推断冒充为本地资料。
- 7B 模型在 8GB 显存下以 4-bit 运行，长上下文和高并发场景需要额外优化。
- 当前验证重点是大模型能否完整读取本地题目，不等同于完整比赛自动答题评测。
- `data/样题.txt` 为比赛样题，仅用于本地验证；如需公开分发，请确认相关授权。
