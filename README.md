# Hospital Agent Baseline Example

这是一个虚拟诊疗比赛的医生 Agent 示例项目，展示如何基于 hospital-agent-sdk 开发、训练、测试和部署自己的医生 Agent。

## 目录说明

```
baseline_example/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # 需要修改：医生 Agent 主逻辑，包含 train/test 和诊疗策略
│   ├── prompt.py         # 需要修改：Prompt 模板和输出格式约束
│   ├── memory.py         # 需要修改：训练反思、病例经验、检索记忆等
│   ├── knowledge.py      # 知识库召回、标准名称规范化、疾病画像检索
│   ├── rag_retriever.py  # Hybrid RAG 统一检索入口
│   ├── exam_strategy.py  # 检查策略 Agent
│   ├── inquiry_strategy.py # 问诊策略 Agent
│   ├── treatment_strategy.py # 治疗策略 Agent
│   ├── memory_system.py  # 轻量结构化记忆管理器
│   └── qc.py             # 提交前质控 Agent
├── data/
│   ├── memory_data/
│   │   └── memory.md     # 可自定义：baseline 默认的本地文件记忆
│   └── ref_data/
│       ├── departments.json
│       ├── diseases_catalog.json
│       ├── examinations_catalog.json
│       ├── disease_profiles.json
│       └── disease_profiles_extra.json
├── config.yaml           # 可配置：训练患者数量、输出目录、memory 路径等
├── train.py              # 可选修改：本地训练入口，默认可直接使用
├── test.py               # 可选修改：本地测试和批量评估示例，默认可直接使用
├── requirements.txt      # 必须维护：新增第三方库需要写在这里
├── Dockerfile            # 一般不需要修改：部署入口，默认启动 python3 -m agent.agent
└── README.md
```

## 重点修改文件

参赛时需要重点修改以下文件：

- **agent/agent.py**：负责诊疗流程和 action 调用策略
- **agent/prompt.py**：负责模型输入和输出格式约束
- **agent/memory.py**：负责训练反思、病例经验、检索记忆等记忆逻辑

## 当前多智能体结构

本项目已经在 baseline 上升级为轻量多智能体编排：

- **主诊断 Agent**：`agent/agent.py`，负责整体流程、Planner 决策和 action 调用。
- **问诊策略 Agent**：`agent/inquiry_strategy.py`，根据疾病画像补齐关键追问和红旗信号。
- **检查策略 Agent**：`agent/exam_strategy.py`，根据候选疾病补齐必查检查并过滤无效检查名。
- **治疗策略 Agent**：`agent/treatment_strategy.py`，根据疾病画像补齐治疗原则、个体化和安全提醒。
- **质控 Agent**：`agent/qc.py`，提交前修复 `diagnosis/treatment_plan/reasoning` 并规范诊断名称。
- **知识库层**：`agent/knowledge.py`，加载标准目录和疾病画像，负责症状召回、标准化和 RAG 上下文。
- **Hybrid RAG 检索层**：`agent/rag_retriever.py`，统一召回疾病画像、标准检查、历史病例经验和策略补丁。
- **结构化记忆层**：`agent/memory_system.py`，统一管理工作记忆、病例经验记忆、语义知识记忆和策略补丁记忆。

## 配置说明

### 环境变量

| 变量名 | 说明 |
|--------|------|
| `SERVICE_BASE_URL` | 比赛后端服务地址 |
| `SERVICE_TRAIN_TOKEN` | 训练阶段访问令牌 |
| `MODEL_API_KEY` | 大语言模型调用密钥 |
| `TEAM_ID` | 队伍账号 |

### config.yaml

- `output_dir`：训练和测试产物输出目录
- `train.selection`：患者选取方式（random/forward/reverse）
- `train.patient_count`：训练患者数量
- `train.random_seed`：随机种子
- `test.selection`：测试患者选取方式
- `test.patient_count`：测试患者数量
- `memory.md_path`：记忆文件路径
- `max_ask_rounds`：问诊轮次上限
- `max_exam_rounds`：检查轮次上限
- `ref_data_dir`：标准疾病、检查、科室和疾病画像目录
- `self_improve_enabled`：是否启用训练后自迭代补丁
- `policy_store_path`：策略补丁存储路径
- `memory_system.working_ttl_seconds`：单病例工作记忆保留时间
- `memory_system.max_working_cases`：同时保留的工作病例数量上限
- `service.endpoint_prefixes`：比赛后端 action 接口前缀兜底；如果服务实际是 `/api/ask_patient`，可设为 `["/api"]`
- `rag.top_k`：Hybrid RAG 最终注入的 chunk 数量
- `rag.enable_mqe`：是否启用规则版多查询扩展
- `rag.candidate_pool_multiplier`：候选池扩展倍数，先扩大召回再去重排序
- `rag.score_threshold`：过滤弱相关 chunk 的最低分数阈值

## Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export SERVICE_BASE_URL=https://baconroot-hospital-service.ms.show
export SERVICE_TRAIN_TOKEN=<your-train-service-token>
export MODEL_API_KEY=<your-model-api-key>
export TEAM_ID=<your-team-id>
```

### 3. 运行本地训练

```bash
python train.py
```

### 4. 启动测试服务

```bash
python -m agent.agent
```

服务默认监听 `0.0.0.0:7860`，测试接口为 `POST /test`。

### 5. 运行完整测试

```bash
python test.py
```

## 医生可用 Actions

| Action | 说明 |
|--------|------|
| `ask_patient` | 向患者提问，返回患者回复 |
| `order_examination` | 申请检查，返回检查结果 |
| `prescribe_treatment` | 提交诊断和治疗方案 |
| `evaluation` | 训练阶段获取评测结果 |
| `batch_evaluation` | 批量评估测试结果 |

## 注意事项

- `data/ref_data/` 是标准科室、疾病和检查名称，不要修改
- 提交结果中的诊断和检查名称必须使用标准名称
- `Dockerfile` 一般不需要修改，以免影响平台部署和评测
- 记忆存储形式不限定，但不要依赖 docker-compose 拉起额外服务
- 自定义疾病画像放在 `disease_profiles*.json`，不改变标准 catalog 文件
- Docker 构建会通过 `.dockerignore` 排除输出目录、缓存和本地冒烟脚本
