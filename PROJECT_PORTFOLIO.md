# 🏥 Hospital Agent — 面向虚拟诊疗比赛的多智能体医生系统

> 一个从"单文件硬编码 Baseline"演进为"LLM 驱动 + 多智能体协作 + 自迭代闭环"的完整参赛作品。

---

## 一、项目定位

面向**虚拟诊疗竞赛平台（ModelScope Studio）**的医生 Agent，负责在容器内 HTTP 服务的形式承接平台派发的患者案例，完成 **问诊 → 检查 → 诊断 → 治疗** 全流程，并接受官方 `batch_evaluation` 评测。

- **交付形态**：Docker 镜像，7860 端口，`POST /test`
- **入口**：`python3 -m agent`
- **评测协议**：HTTP 响应体顶层需含 `final_result` / `final_results` 字段

---

## 二、技术栈

| 层次 | 选型 | 说明 |
|---|---|---|
| 运行时 | Python 3.10-slim | 与官方部署环境一致 |
| Web 服务 | `aiohttp` | 全异步，避免 LLM 长请求阻塞 |
| HTTP 客户端 | `httpx` (async) | 调平台后端 & LLM API |
| LLM 接入 | **自研极简客户端** (`agent/llm.py`) | 直接走 OpenAI 兼容协议，不引入 `openai` SDK，减少镜像体积 |
| 模型 | 通义千问 `qwen-plus`（DashScope）| 可通过环境变量热切换 base_url / model_name |
| 配置 | `pyyaml` + `config.yaml` + ENV 变量 | ENV 覆盖 YAML，YAML 覆盖默认值 |

---

## 三、系统架构

```
                       ┌────────────────────────────────────┐
                       │      aiohttp Server  :7860         │
                       │       POST /test                   │
                       └────────────────┬───────────────────┘
                                        │
                                ┌───────▼────────┐
                                │  MyDoctorAgent │  ← 主诊断 Agent (Planner)
                                └───┬────────┬───┘
              ┌─────────────┬──────┘        └──────┬────────────┐
              ▼             ▼                      ▼            ▼
     ┌───────────────┐┌───────────────┐   ┌───────────────┐┌──────────────┐
     │ Inquiry Agent ││  Exam  Agent  │   │ Treatment Ag. ││ Quality Ag.  │
     │ 追问红旗信号  ││ 补齐必查过滤 │   │ 个体化+安全   ││ 提交前质控   │
     └───────────────┘└───────────────┘   └───────────────┘└──────────────┘
              │             │                      │            │
              └─────────────┴──────┬───────────────┴────────────┘
                                   ▼
                         ┌────────────────────┐
                         │  KnowledgeBase     │  疾病画像 RAG + 标准名称规范化
                         └────────────────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ HybridRAGRetriever │  MQE + 多路 chunk 合并排序
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ DoctorAgentMemory  │  Working / Episodic / Semantic / Policy
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  DoctorMemory      │  病例经验 JSON 存储 + 多维检索
                         └────────────────────┘

                    ── 训练后自迭代闭环 ──
       ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
       │ DefectDetector│→ │ PolicyStore  │→ │ ShadowReplay │
       │  规则+LLM归因 │   │ 策略补丁沉淀 │   │ 影子重放验证 │
       └──────────────┘   └──────────────┘   └──────────────┘
```

---

## 四、核心亮点

### 1️⃣ 规划-执行-反思（PER）架构
- [`Planner`](agent/agent.py:75)：制定全局策略 + 决策下一步 action
- **Phase 状态机**：`INITIAL → INQUIRY → EXAMINATION → DIAGNOSIS → TREATMENT → COMPLETED`
- **ActionType**：`ask_patient` / `order_examination` / `prescribe_treatment` / `replan`
- 支持**动态重规划**（信息不足时回退阶段）

### 2️⃣ 多智能体协作
5 个专职 Agent + 1 个知识库层，各司其职、可插拔：
| 组件 | 文件 | 职责 |
|---|---|---|
| 主 Agent | [`agent/agent.py`](agent/agent.py) | Planner + Executor 编排 |
| 问诊策略 | [`agent/inquiry_strategy.py`](agent/inquiry_strategy.py) | 关键症状追问 + 红旗信号 |
| 检查策略 | [`agent/exam_strategy.py`](agent/exam_strategy.py) | 必查项补齐 + 非法名过滤 |
| 治疗策略 | [`agent/treatment_strategy.py`](agent/treatment_strategy.py) | 个体化 + 安全提醒 |
| 质控 Agent | [`agent/qc.py`](agent/qc.py) | 提交前 diagnosis / treatment_plan / reasoning 规范化 |
| 知识库 | [`agent/knowledge.py`](agent/knowledge.py) | 疾病画像 RAG + 标准名称对齐 |
| Hybrid RAG | [`agent/rag_retriever.py`](agent/rag_retriever.py) | 统一召回疾病画像、标准检查、病例经验和策略补丁 |

### 3️⃣ 结构化记忆系统
- [`DoctorAgentMemory`](agent/memory_system.py)：统一调度工作记忆、病例经验记忆、语义知识记忆和策略补丁记忆
- `WorkingCaseMemory`：保存单病例的问诊历史、已采集信息、检查结果和候选诊断，带 TTL 与容量上限
- `EpisodicMemoryAdapter`：复用 [`DoctorMemory`](agent/memory.py) 的历史病例经验检索
- `SemanticMemoryAdapter`：复用 [`KnowledgeBase`](agent/knowledge.py) 的疾病画像和标准目录
- `PolicyMemoryAdapter`：复用 [`PolicyStore`](agent/policy_store.py) 的训练后策略补丁
- [`HybridRAGRetriever`](agent/rag_retriever.py)：统一输出 `disease_profile` / `standard_exam` / `case_experience` / `policy_patch` 四类 chunk
- 规则版 MQE 会把口语主诉扩展为标准症状和候选疾病，例如“胸口闷、喘不上气”扩展到胸闷、胸痛、呼吸困难、心悸、冠心病、心肌梗死、肺炎等方向
- 支持候选池扩展 `candidate_pool_multiplier` 和最低分过滤 `score_threshold`
- 结构化 JSON 存储，从旧 Markdown 自动迁移
- **多维检索**：症状关键词 / 诊断模式 / 质量分 / 时间衰减
- **低分病例优先保留**：从失败中学习
- 单条上限 & 总条数上限双维度控制

### 4️⃣ 自迭代闭环（差异化亮点）
训练结束后：
- [`DefectDetector`](agent/critic.py) 用「规则通道 + LLM 通道」双路识别缺陷
- [`PolicyStore`](agent/policy_store.py) 把缺陷沉淀为**策略补丁**（可回滚）
- [`ShadowReplay`](agent/replay.py) 影子重放验证补丁效果，未增益不入库

配置开关：`self_improve_enabled: true`

### 5️⃣ 评测器字段兼容层（工程细节）
针对官方 `batch_evaluation` 服务端的严苛字段契约：
- [`_default_final_result`](agent/server.py:33)：兜底结构，保证 `KeyError` 免疫
- [`_normalize_final_result`](agent/server.py:53)：兼容 `diagnosis` / `diagnoses` / `finalDiagnosis` / `plan` 多种命名
- [`_extract_patient_ids`](agent/server.py) ：兼容 `patient_id` / `caseId` / `cases[]` / `input.patient_ids` 多种入参
- 在 `/test` 响应**顶层同时暴露** `final_result` + `final_results`，兼容单例 & 批量口径

### 6️⃣ 全链路降级
每个 LLM 方法都有**规则回退**，LLM 挂了服务依然可用；敏感字段（token / api_key）走 [`_redact_for_log`](agent/server.py) 脱敏。

---

## 五、可配置项（[`config.yaml`](config.yaml)）
- 训练/测试患者选取（`random` / `forward` / `reverse`）
- LLM 参数（base_url / model_name / temperature / retries）
- 问诊 & 检查轮次上限（`max_ask_rounds` / `max_exam_rounds`）
- 记忆容量（`max_notes` / `max_note_chars`）
- 工作记忆 TTL 和并发病例上限（`memory_system.*`）
- Hybrid RAG 检索策略（`rag.top_k` / `rag.enable_mqe` / `rag.score_threshold`）
- 自迭代总开关（`self_improve_enabled`）

---

## 六、部署

```bash
# 本地
pip install -r requirements.txt
python -m agent          # 7860 端口

# 生产
docker build -t hospital-agent .
docker run -p 7860:7860 \
  -e MODEL_API_KEY=... \
  -e SERVICE_BASE_URL=... \
  -e SERVICE_TRAIN_TOKEN=... \
  -e TEAM_ID=... \
  hospital-agent
```

---

## 七、工程沉淀

| 议题 | 处置 |
|---|---|
| 单文件 Baseline 硬编码 | 拆成 6 个可测试单元 + 依赖注入 |
| 官方 batch_evaluation KeyError | 增加字段兼容层 + 顶层字段注入 |
| 官方后端偶发 404 | Agent 侧记录证据链，不吞异常 |
| LLM 请求超时 | 异步 + 指数退避重试 + 规则降级 |
| 敏感配置泄露风险 | `_SENSITIVE_KEYS` 白名单脱敏 |

---

## 八、目录一览

```
project/
├── agent/                  # 核心多智能体
│   ├── agent.py            # Planner + Executor 主逻辑
│   ├── llm.py              # 极简 OpenAI 兼容客户端
│   ├── prompt.py           # 9 个 Prompt 模板
│   ├── memory.py           # 多维检索记忆
│   ├── memory_system.py    # 统一记忆管理器
│   ├── knowledge.py        # 知识库 & RAG
│   ├── rag_retriever.py    # Hybrid RAG 统一检索
│   ├── {inquiry,exam,treatment}_strategy.py
│   ├── qc.py               # 质控
│   ├── critic.py           # 缺陷识别
│   ├── policy_store.py     # 策略补丁
│   ├── replay.py           # 影子重放
│   └── server.py           # aiohttp 服务 + 字段兼容层
├── hospital_agent/         # 官方 SDK 基类
├── data/
│   ├── memory_data/        # 运行时记忆
│   └── ref_data/           # 标准疾病 / 检查 / 科室目录
├── config.yaml
├── Dockerfile
├── train.py / test.py      # 本地训练 & 测试入口
└── requirements.txt
```

---

## 九、一句话总结
> **从"跑通比赛"到"跑赢比赛"**：本项目把一个单文件 Baseline 演进为一个具备**多智能体协作、知识库 RAG、多维记忆检索、训练后自迭代**的完整临床决策系统，并针对评测协议做了工业级的字段兼容与降级设计。
