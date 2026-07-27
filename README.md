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
│   ├── clinical_evidence.py # 通用临床证据标准化
│   ├── diagnosis_engine.py  # 全目录证据评分与终诊裁决
│   ├── diagnosis_critic.py  # 提交前确定性/受限 LLM 审查
│   ├── diagnostic_learning.py # shadow 诊断规则学习
│   ├── treatment_safety.py # 治疗禁忌与过敏安全门
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
├── Dockerfile            # 一般不需要修改：部署入口，默认启动 python3 -m agent
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

## 证据优先诊断链

当前终诊路径不再由 LLM 单独决定：

```text
Planner 问诊/检查
  -> ClinicalEvidenceNormalizer 按字段与局部语义生成 EvidenceBundle
  -> Hybrid RAG 召回疾病画像、标准检查、历史经验和策略补丁
  -> LLM 开放提出具体临床候选
  -> OpenWorldDiagnosisResolver 执行别名、上下位和保守模糊匹配
  -> DiagnosisDecisionEngine 对 50 个官方疾病和 21 个受控扩展全量评分
  -> DiagnosisCritic 检查低置信、硬反证、遗漏病因和未解释严重异常
  -> QualityAgent 校验名称边界
  -> TreatmentStrategyAgent 读取诊断协议
  -> TreatmentSafetyGate 过滤过敏、年龄和检查异常相关冲突
  -> prescribe_treatment
```

LLM 可以提出目录外候选，但原始名称不能直接提交。只有映射到官方 catalog 或 evaluation 已确认的受控扩展后，才会进入证据评分；无法可靠映射的候选只保留在审计与学习记录中。反思与 evaluation 反馈只生成 pending/shadow 候选，不直接改写正式知识。

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
- `service.invoke_path`：当前比赛服务使用 `/invoke` 入口承载患者问诊。
- `service.exam_results_path` / `service.case_evaluation_path`：当前比赛服务分别使用 `/exam/results` 获取检查结果、`/evaluate/case` 获取单病例训练评估。
- `rag.top_k`：Hybrid RAG 最终注入的 chunk 数量
- `rag.enable_mqe`：是否启用规则版多查询扩展
- `rag.candidate_pool_multiplier`：候选池扩展倍数，先扩大召回再去重排序
- `rag.score_threshold`：过滤弱相关 chunk 的最低分数阈值
- `diagnosis.trusted_threshold`：可信诊断最低分，默认 `0.65`
- `diagnosis.open_world_candidates`：允许 LLM 提出目录外具体候选，再执行受控标准化
- `diagnosis.name_match_threshold`：名称模糊映射最低相似度，默认 `0.84`
- `diagnosis.name_match_margin`：最佳与次佳映射的最小差值，避免歧义名称自动映射
- `diagnosis.margin_threshold`：前两名小于该差值时触发低边际审查
- `diagnosis.max_final_diagnoses`：最终最多提交的诊断数
- `final_critic.max_llm_calls`：每例终诊 Critic 的最大 LLM 调用数
- `execution.case_timeout_seconds`：单病例总预算，默认 `235` 秒；训练时会从中预留 evaluation 和 Reflection 时间
- `learning.freeze_active_knowledge`：训练时冻结正式知识与 active 策略
- `learning.auto_promote_exam_aliases`：是否允许检查别名自动晋级，默认关闭

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

每次训练会生成：

```text
outputs/train/<run_id>/training_results.jsonl
outputs/train/<run_id>/training_summary.json
```

冻结代码与正式知识后，依次运行 seed 46、47、48：

```bash
python tools/run_frozen_training.py --seeds 46 47 48 --patient-count 5
```

汇总报告包含诊断准确率、检查精确率、治疗评分与安全性、候选 Recall@5、Critic 触发率、平均耗时和超时数。

### 4. 启动测试服务

```bash
python -m agent
```

服务默认监听 `0.0.0.0:7860`，测试接口为 `POST /test`。

### 5. 运行完整测试

```bash
python test.py
```

### 6. 离线回放和单元测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tools/run_diagnostic_replay.py tests/fixtures/diagnostic_replay_cases.jsonl --strict
```

离线回放不访问比赛后端或模型服务，可先验证否定语义、证据评分、名称合法性和已知问题病例。

## 知识来源与许可

- 结构化知识记录来源 URL、版本和审核状态；运行时不联网。
- [WHO Guidelines](https://www.who.int/publications/who-guidelines)作为权威指南索引。
- 血管炎规则记录了[2021 ACR/VF ANCA 相关血管炎指南](https://rheumatology.org/vasculitis-guideline)的版本元数据。
- [ESC 瓣膜病指南页面](https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/)声明软件或 AI 转化需要正式许可，因此当前规则库明确标记为未获许可时不转化其内容。

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
