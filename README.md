# MedicalAI

多智能体医疗问答系统，基于 LangGraph 编排。后端 FastAPI + 前端 React，本地嵌入模型与 ChromaDB 向量检索，通过 Plan-Replan 闭环与独立事实核查实现可靠的医疗问答。

## 技术栈

| 类别 | 技术 |
|------|------|
| 编排框架 | LangGraph |
| LLM | DashScope 通义千问 (qwen3-max) |
| 嵌入模型 | 本地 HuggingFace (bge-small-zh-v1.5) |
| 向量数据库 | ChromaDB (cosine, MMR 检索) |
| Reranker | 本地 CrossEncoder (bge-reranker-base) |
| 后端 | FastAPI + SQLAlchemy + SQLite |
| 前端 | React 19 + Vite 7 + Tailwind CSS 4 + daisyUI 5 |
| 部署 | Docker Compose |

## 快速开始

以下命令在 **PowerShell** 中执行。

### 1. 环境准备

```
Python 3.10+
Node.js 18+
```

复制 `.env.example` 为 `.env`，填入你的 API Key：


```powershell
cp .env.example .env
```

然后编辑 `.env`，填入真实 Key：

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. 下载嵌入模型

项目使用本地嵌入模型，需先下载：

```powershell
# 安装依赖
cd backend
pip install -r requirements.txt

# 下载 bge-small-zh-v1.5 嵌入模型和 bge-reranker-base 重排序模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
```

也可设置镜像加速（PowerShell）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 3. 运行

一键启动（后端 + 前端）：

```powershell
python run.py
```

或分别启动：

```powershell
# 后端
cd backend; python -m uvicorn app.main:app --reload --port 8000

# 前端
cd frontend; npm install; npm run dev
```

- API 文档：`http://localhost:8000/docs`
- 前端界面：`http://localhost:5173`

### Docker 部署

```powershell
docker compose up --build
```

## 工作流架构

系统由 5 个 LangGraph Agent 构成，通过 Plan-Replan 闭环与独立事实核查机制协同工作：

```
用户消息 → MemoryAgent → QueryRewriterAgent → PlannerAgent
                                                    │
                                          ┌─────────┴──────────┐
                                          │                    │
                                     ResearchAgent        CriticAgent
                                     (ReAct 循环)       (事实核查)
                                          │                    │
                                          └─────────┬──────────┘
                                                    │
                                              返回结果
```

- **MemoryAgent** — 加载短期对话历史与长期用户画像（SQLite 持久化）
- **QueryRewriterAgent** — 意图识别（7 分类）、口语→术语规范化、生成扩展查询词
- **PlannerAgent** — 初始路由决策（RAG / 工具 / LLM 直答）+ 执行结果评估，不满足时触发重规划
- **ResearchAgent** — ReAct 循环（THINK→ACT→OBSERVE），自主选择 RAG 检索、工具查询、Wikipedia、Tavily 搜索
- **CriticAgent** — 独立事实核查，通过 PubMed（可选）或 RAG 文档验证答案，检测幻觉

### 关键设计

- **Plan-Replan** — Planner 评估 Research 输出质量，不满足则注入指令重执行（上限 1 次）
- **Early-Exit** — Reranker 评分 ≥ 0.85 时跳过 LLM 决策，直接输出
- **盲区检测** — 首次 RAG 检索快速检测知识库覆盖，盲区直接路由到 Tavily
- **第三方核查** — CriticAgent 使用独立数据源（PubMed / RAG 文档），与 ResearchAgent 隔离

## 项目结构

```
backend/
├── app/
│   ├── agents/           # 5 个 LangGraph Agent
│   │   ├── memory.py
│   │   ├── query_rewriter.py
│   │   ├── planner.py
│   │   ├── research.py
│   │   └── critic.py
│   ├── api/v1/           # REST API 端点
│   ├── core/             # 配置、工作流、状态定义
│   │   ├── config.py
│   │   ├── langgraph_workflow.py
│   │   └── state.py
│   ├── db/               # SQLAlchemy 会话管理
│   ├── memory/           # 短期/长期记忆
│   ├── models/           # ORM 模型
│   ├── schemas/          # Pydantic 模式
│   ├── services/         # 业务逻辑
│   ├── tools/            # 工具层
│   │   ├── llm_client.py
│   │   ├── vector_store.py
│   │   ├── reranker.py
│   │   ├── tavily_search.py
│   │   ├── wikipedia_search.py
│   │   └── ...
│   ├── main.py           # FastAPI 入口
│   └── ...
├── tests/                # pytest 测试套件
├── Dockerfile
├── .dockerignore
└── requirements.txt

frontend/
├── src/
│   ├── App.jsx
│   ├── index.jsx
│   └── index.css
├── Dockerfile
├── nginx.conf
├── .dockerignore
├── package.json
└── vite.config.js
```

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 通义千问 LLM API Key | 是 |
| `TAVILY_API_KEY` | Tavily 联网搜索 Key | 否 |
| `EMBEDDING_MODEL` | 本地嵌入模型路径 | 否（自动探测） |
| `RERANKER_MODEL` | 本地 Reranker 路径 | 否（自动探测） |
| `RERANKER_TOP_K` | 精排后文档数（默认 5） | 否 |
| `MCP_ENABLED` | 启用 MCP 外部工具（true/false，默认 true） | 否 |

## 测试

```powershell
cd backend
python -m pytest tests/ --cov=app tests/

cd frontend
npm run test
```
