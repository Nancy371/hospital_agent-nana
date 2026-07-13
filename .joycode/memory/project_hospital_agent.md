---
name: project_hospital_agent
description: Hospital Agent Baseline Example 项目架构和部署修复记录
type: project
---

## 项目概述
虚拟诊疗比赛的医生 Agent 示例项目，部署在 ModelScope Studio Docker 容器，7860 端口 HTTP 服务。

## 关键架构决策
- **LLM 驱动升级**（2026-07-03）：6 个硬编码方法全部改为 LLM 驱动 + 记忆注入
- **LLM 客户端**：agent/llm.py，使用 httpx 直接调用 OpenAI 兼容 API（非 openai SDK），减少依赖
- **记忆系统**：agent/memory.py，JSON 存储 + 多维检索（症状/诊断/质量/时间衰减）+ 低分病例优先保留
- **Prompt 工程**：agent/prompt.py，9 个模板 + 经验注入 + 结构化 JSON 输出
- **降级策略**：每个 LLM 方法都有规则回退，保证服务可用性

## 环境变量
- MODEL_API_KEY（必需）：LLM API 密钥
- MODEL_BASE_URL：默认 https://dashscope.aliyuncs.com/compatible-mode/v1
- MODEL_NAME：默认 qwen-plus
- SERVICE_BASE_URL / SERVICE_TRAIN_TOKEN / TEAM_ID：比赛平台配置

## 已验证
- 所有 4 个模块语法检查通过
- LLMClient / DoctorPrompt / DoctorMemory / MyDoctorAgent 导入正常
- MyDoctorAgent 10 个关键方法全部存在