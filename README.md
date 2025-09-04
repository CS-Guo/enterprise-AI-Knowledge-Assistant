# 企业智能知识助手 (Enterprise AI Knowledge Assistant)

基于 LangGraph + RAG + MCP 架构的企业知识助手，提供知识检索、智能对话、文档管理与自动化工具（邮件、日历、文件）的统一入口。后端使用 FastAPI 提供 API，前端使用 Streamlit 提供现代化交互界面，向量存储使用 ChromaDB 持久化管理。


## 功能特性
- 智能对话：面向企业知识的问答与多轮对话
- 文档管理：上传文档、分块处理、嵌入入库、相似度检索
- RAG 检索：基于向量检索的相关内容召回，支持简单的查询增强与阈值过滤
- 工具中心（MCP）：
  - 邮件工具：模板填充与发送（安全示例）
  - 日历工具：事件创建、排班示例
  - 文件工具：本地文件搜索/读取/写入
- 系统监控：组件状态与向量库统计
- 现代化前端：基于 Streamlit 的多标签页 UI（对话、文档、工具、信息面板）


## 架构概览
- 后端 API：FastAPI（`/src/api/main.py`, `/src/api/routes.py`）
  - 文档：OpenAPI /docs 与 /redoc
  - 路由聚合：`/api/v1` 前缀
- Agent 与工作流：`/src/agents/knowledge_agent.py`, `/src/agents/workflow.py`
- RAG 子系统：
  - 文档处理：`/src/rag/document_processor.py`
  - 检索器：`/src/rag/retriever.py`
  - 向量库：`/src/rag/vector_store.py`（Chroma 持久化，OpenAIEmbeddings 生成向量）
- MCP 工具：`/src/mcp/*.py`（邮件、日历、文件）
- 前端：Streamlit 界面 `src/frontend/app.py`
- 配置：`config/settings.py`（支持 `.env` 注入）


## 目录结构（节选）
```
enterprise-AI-Knowledge-Assistant/
├── config/
│   └── settings.py
├── src/
│   ├── agents/
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   ├── frontend/
│   │   └── app.py
│   ├── mcp/
│   └── rag/
└── tests/
```


## 快速开始

### 1. 环境要求
- Python 3.9+（推荐 3.10/3.11）
- macOS/Linux/Windows 任一环境

### 2. 克隆与进入目录
```
# 使用你的 GitHub 仓库地址
git clone git@github.com:CS-Guo/enterprise-AI-Knowledge-Assistant.git
cd enterprise-AI-Knowledge-Assistant
```

### 3. 创建虚拟环境并安装依赖
（项目未附带 requirements.txt，可按下列依赖安装）
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -U pip
pip install fastapi uvicorn pydantic-settings streamlit pandas requests \
            chromadb sentence-transformers langchain-openai
```
如使用测试，请额外安装：
```
pip install pytest
```

### 4. 配置环境变量（.env）
项目通过 `pydantic-settings` 从根目录 `.env` 读取配置。请创建 `.env` 文件（已在 .gitignore 中忽略）：
```
# 大模型/嵌入配置（示例）
OPENAI_API_KEY="YOUR_API_KEY"
BASE_URL="https://api.your-llm-provider.com/v1"
OPENAI_MODEL="gpt-4o-mini"              # 或你的对话模型
EMBEDDING_MODEL="text-embedding-3-small" # 或你的嵌入模型

# 向量数据库/数据目录
VECTOR_DB_PATH="./vectordb"
DOCUMENTS_PATH="./data/documents"

# API 服务配置
API_HOST="0.0.0.0"
API_PORT=8000

# Agent 推理参数
MAX_ITERATIONS=10
TEMPERATURE=0.1

# 邮件配置（如需启用）
EMAIL_SENDER="you@example.com"
EMAIL_PASSWORD="your_app_password"
EMAIL_SMTP_SERVER="smtp.example.com"
EMAIL_SMTP_PORT=587
DEFAULT_RECIPIENT="someone@example.com"
```
重要：请勿在代码中硬编码密钥、密码或任何敏感信息，统一放入 `.env`。

### 5. 启动后端 API
方式 A（推荐）：
```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
方式 B：
```
python src/api/main.py
```
API 文档：
- Swagger UI: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

### 6. 启动前端（Streamlit）
```
streamlit run src/frontend/app.py
```
默认前端将访问 `http://localhost:8000/api/v1` 后端。


## 使用说明（前端）
- 智能对话：在“💬 智能对话”中直接提问，系统结合向量检索返回答案
- 文档管理：在“📚 文档管理”中上传文档，系统会分块并入库，可查看统计信息
- 工具中心：在“🔧 功能列表”中查看可用工具与参数说明（邮件、日历、文件）
- 系统状态：侧边栏展示系统与组件运行状态


## RAG/向量库说明
- 向量库：ChromaDB 持久化在 `VECTOR_DB_PATH`
- 嵌入模型：默认使用 `langchain_openai.OpenAIEmbeddings`（由 `.env` 中的 `OPENAI_API_KEY`、`BASE_URL` 与 `EMBEDDING_MODEL` 控制）
- 检索器：`src/rag/retriever.py` 提供查询增强、过滤阈值、可选重排序的示例实现


## API 概览
- 基础路径：`/api/v1`
- 示例能力：系统状态、工具列表、文档上传/检索等（具体见 `src/api/routes.py` 与在线文档 `/docs`）


## 测试
运行示例测试（安全版，仅本地行为，不执行真实外发）：
```
pytest -q
# 或仅运行工具测试
pytest tests/test_all_tools.py -q
```
