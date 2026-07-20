"""
医生 Agent 主逻辑模块。

基于 LLM 驱动的智能诊疗流程，采用规划-执行-反思架构：
- 规划(Planner)：全局诊疗策略制定 + 阶段转换 + 动态重规划
- 执行(Executor)：问诊/检查/诊断/治疗等具体操作
- 反思(Reflection)：自我批判 + 策略调整 + 经验积累
- 记忆(Memory)：历史经验检索注入 + 低分病例重点学习

Planner 作为中枢，通过 Reflection/Criticism 机制将宏观目标分解为具体步骤，
LLM 进行深度推理后决策下一步操作（tool call）。
"""

import asyncio
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from hospital_agent import BaseDoctorAgent
from .llm import LLMClient
from .prompt import DoctorPrompt
from .memory import DoctorMemory
from .knowledge import KnowledgeBase
from .critic import DefectDetector
from .memory_system import DoctorAgentMemory
from .policy_store import PolicyStore
from .exam_strategy import ExamStrategyAgent
from .inquiry_strategy import InquiryStrategyAgent
from .qc import QualityAgent
from .treatment_strategy import TreatmentStrategyAgent
from .structural_diagnosis import StructuralDiagnosisAgent
from .evidence_engine import EvidenceDiagnosisEngine
from .clinical_evidence import ClinicalEvidenceNormalizer, EvidenceAgent, EvidenceBundle
from .diagnosis_engine import DiagnosisDecisionEngine
from .diagnosis_critic import DiagnosisCritic
from .diagnostic_learning import DiagnosticLearningStore
from .treatment_safety import TreatmentSafetyGate

logger = logging.getLogger(__name__)


def _overall_score(report: Dict[str, Any]) -> Optional[float]:
    """从平台评估报告中提取综合分。取三项均值作为综合指标。"""
    if not report:
        return None
    keys = ("diagnosisAccuracy", "examinationPrecision", "treatmentOverallScore")
    vals = []
    for k in keys:
        v = report.get(k)
        try:
            if v is not None:
                vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


# ============ 诊疗阶段枚举 ============

class Phase(str, Enum):
    """诊疗阶段状态机。"""
    INITIAL = "initial"        # 初始接诊
    INQUIRY = "inquiry"        # 问诊阶段
    EXAMINATION = "examination"  # 检查阶段
    DIAGNOSIS = "diagnosis"    # 诊断阶段
    TREATMENT = "treatment"    # 治疗阶段（提交方案）
    COMPLETED = "completed"    # 诊疗完成


class ActionType(str, Enum):
    """可执行的操作类型，对应 tool call。"""
    ASK_PATIENT = "ask_patient"           # 询问患者
    ORDER_EXAMINATION = "order_examination"  # 申请检查
    PRESCRIBE_TREATMENT = "prescribe_treatment"  # 提交诊疗方案
    REPLAN = "replan"                     # 重新规划


# ============ 规划器 ============

class Planner:
    """诊疗规划器 —— 中枢模块。

    职责：
    1. 制定全局诊疗策略（_plan）
    2. 决策下一步具体操作（decide_next_action）
    3. 阶段转换判断与执行
    4. 通过 Reflection/Criticism 机制自我修正策略
    5. 与 Memory 交互整合历史信息

    架构：
    Planner → LLM(深度推理) → Action(tool call)
         ↑                        |
         └── Reflection/Criticism ─┘
    """

    def __init__(self, prompt: DoctorPrompt, llm_chat_json, llm_chat, memory: DoctorMemory):
        """初始化规划器。

        Args:
            prompt: Prompt 模板管理器
            llm_chat_json: 异步 LLM JSON 调用函数
            llm_chat: 异步 LLM 文本调用函数
            memory: 记忆管理器
        """
        self.prompt = prompt
        self._llm_chat_json = llm_chat_json
        self._llm_chat = llm_chat
        self.memory = memory

        # 规划状态
        self.current_phase = Phase.INITIAL
        self.current_plan: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.criticism_result: Optional[Dict[str, Any]] = None

        # 阶段计数器
        self.inquiry_rounds = 0
        self.exam_rounds = 0

        # 配置
        self.max_inquiry_rounds = 5
        self.max_exam_rounds = 3
        self.max_total_actions = 15  # 安全上限，防止无限循环

        # 批判频率控制（成本优化）：仅前 N 次 plan 触发 criticism，后续 plan 跳过。
        # 因为初始规划质量最关键，后续多为微调，无需每次都走 2 次 LLM。
        self._plan_call_count = 0
        self.criticism_max_calls = 2

        # 跨患者软复用的教训种子（由 soft_reset 填充）
        self._carry_lessons: Optional[str] = None

        # 策略补丁库（由外部注入；None 时不启用）
        self.policy_store = None  # type: ignore[assignment]
        # 当前 plan 命中的补丁 ID 列表（供反思阶段 record_outcome 使用）
        self._last_used_patch_ids: List[str] = []

    def _record_action(self, action_type: str, target: str, result_summary: str) -> None:
        """记录已执行的操作到历史。"""
        self.action_history.append({
            "type": action_type,
            "target": target,
            "result_summary": result_summary[:200],  # 截断防过长
            "phase": self.current_phase.value,
        })

    def soft_reset(self, keep_lessons: bool = True) -> None:
        """跨患者软复用：重置流程状态但保留可迁移的教训。

        清理项（下一位患者必须干净开始）:
            - current_phase / current_plan / action_history / criticism_result
            - inquiry_rounds / exam_rounds / _plan_call_count
        保留项（跨患者可复用的知识）:
            - self.memory 由外部单例持有，天然保留
            - self.max_* 配置
            - 若 keep_lessons=True 则保留最近一次 plan 的 lessons_learned 摘要
        """
        # 抽取上一位患者留下的教训（若有）
        lessons_carry = None
        if keep_lessons and self.current_plan:
            lessons_carry = self.current_plan.get("lessons_learned") or \
                            self.current_plan.get("strategy", {}).get("lessons_learned")

        # 重置流程状态
        self.current_phase = Phase.INITIAL
        self.current_plan = None
        self.action_history = []
        self.criticism_result = None
        self.inquiry_rounds = 0
        self.exam_rounds = 0
        self._plan_call_count = 0

        # 将教训作为轻量种子挂在 planner 上，下一次 plan() 时可注入 prompt
        self._carry_lessons = lessons_carry if lessons_carry else None
        logger.info(
            f"[规划] 软复用完成，教训种子={'有' if self._carry_lessons else '无'}"
        )

    async def plan(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """制定或更新全局诊疗策略。

        这是规划器的核心方法，通过 LLM 深度推理生成策略规划，
        然后通过 Reflection/Criticism 机制进行自我审查和修正。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            规划结果字典
        """
        # 1. 调用 LLM 生成策略规划
        planning_prompt = self.prompt.build_planning_prompt(
            collected_info=collected_info,
            exam_results=exam_results,
            chat_history=chat_history,
            phase=self.current_phase.value,
            relevant_experience=relevant_experience,
            previous_plan=self.current_plan,
        )

        # 1.5 自迭代增强：拼接 (a) 教训种子 lessons  (b) 命中的策略补丁
        self._last_used_patch_ids = []
        _augment_parts: List[str] = []
        if self._carry_lessons:
            _augment_parts.append("【历史教训（跨患者复用）】\n" + str(self._carry_lessons))
        if self.policy_store is not None:
            try:
                _cands_for_match = []
                _dd = (self.current_plan or {}).get("differential_diagnoses") or []
                for _it in _dd:
                    if isinstance(_it, dict) and _it.get("disease"):
                        _cands_for_match.append(_it["disease"])
                    elif isinstance(_it, str):
                        _cands_for_match.append(_it)
                _ph = (self.current_plan or {}).get("primary_hypothesis")
                if _ph:
                    _cands_for_match.append(_ph)
                _hits = self.policy_store.match(
                    collected_info=collected_info,
                    candidate_diseases=_cands_for_match,
                    include_shadow=False,
                )
                if _hits:
                    _augment_parts.append(self.policy_store.render_for_prompt(_hits))
                    self._last_used_patch_ids = [p.get("id") for p in _hits if p.get("id")]
                    logger.info(
                        f"[规划] 命中策略补丁 {len(_hits)} 项: "
                        f"{[p.get('type') for p in _hits]}"
                    )
            except Exception as _e:
                logger.debug(f"[规划] 补丁匹配失败: {_e}")
        if _augment_parts:
            planning_prompt = "\n\n".join(_augment_parts) + "\n\n" + planning_prompt

        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": "请制定诊疗策略规划。"},
        ]

        plan_result = await self._llm_chat_json(messages, temperature=0.3)

        if not plan_result or "strategy" not in plan_result:
            logger.warning("[规划] LLM 规划失败，使用回退策略")
            plan_result = self._fallback_plan(collected_info, exam_results)

        # 2. Reflection/Criticism：自我批判审查（限流：仅前 N 次 plan 触发，节省成本）
        self._plan_call_count += 1
        criticism = None
        if self._plan_call_count <= self.criticism_max_calls:
            criticism = await self._reflect_and_criticize(
                plan_result, collected_info, exam_results
            )
        else:
            logger.info(
                f"[规划] 已达批判限流阈值({self.criticism_max_calls}次)，本次跳过 criticism 以节省 LLM 调用"
            )

        # 3. 根据批判结果修正计划
        if criticism and criticism.get("overall_assessment") == "needs_replan":
            logger.info("[规划] 批判建议重新规划，执行重规划")
            # 融合批判建议后重新规划
            revised_plan = self._apply_criticism_to_plan(plan_result, criticism)
            plan_result = revised_plan
        elif criticism and criticism.get("overall_assessment") == "needs_adjustment":
            logger.info("[规划] 批判建议微调，应用修正")
            plan_result = self._apply_criticism_to_plan(plan_result, criticism)

        # 4. 更新状态
        self.current_plan = plan_result
        self.criticism_result = criticism

        # 5. 同步阶段状态
        strategy = plan_result.get("strategy", {})
        plan_phase = strategy.get("current_phase", "")
        if plan_phase and plan_phase in [p.value for p in Phase]:
            self.current_phase = Phase(plan_phase)

        # 日志
        hypothesis = plan_result.get("primary_hypothesis", "未知")
        confidence = plan_result.get("hypothesis_confidence", 0)
        priority_actions = strategy.get("priority_actions", [])
        logger.info(f"[规划] 主要假设: {hypothesis} (置信度: {confidence})")
        logger.info(f"[规划] 当前阶段: {self.current_phase.value}")
        if priority_actions:
            first_action = priority_actions[0]
            logger.info(f"[规划] 首要行动: {first_action.get('action')} - {first_action.get('target')}")

        return plan_result

    async def _reflect_and_criticize(
        self,
        current_plan: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """对当前计划进行反思批判。

        通过 LLM 扮演"内在批判者"角色，从假设偏差、信息盲区、
        行动优先级、风险遗漏、阶段转换、认知偏差等维度审查计划。

        Args:
            current_plan: 当前诊疗计划
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果

        Returns:
            批判结果字典，包含 criticisms、plan_revision、overall_assessment
        """
        # 获取最近的思考结果
        thinking_result = None
        if self.current_plan and "differential_diagnoses" in self.current_plan:
            thinking_result = {
                "differential_diagnosis": self.current_plan.get("differential_diagnoses", []),
                "key_unknowns": self.current_plan.get("strategy", {}).get("info_gaps", []),
            }

        criticism_prompt = self.prompt.build_reflection_criticism_prompt(
            current_plan=current_plan,
            collected_info=collected_info,
            exam_results=exam_results,
            thinking_result=thinking_result,
            action_history=self.action_history,
        )
        messages = [
            {"role": "system", "content": criticism_prompt},
            {"role": "user", "content": "请对当前诊疗策略进行批判性审查。"},
        ]

        result = await self._llm_chat_json(messages, temperature=0.4)

        if result and "criticisms" in result:
            high_severity = [c for c in result.get("criticisms", []) if c.get("severity") == "high"]
            assessment = result.get("overall_assessment", "on_track")
            confidence = result.get("confidence_in_plan", 0.5)
            logger.info(f"[批判] 评估: {assessment}, 计划置信度: {confidence}")
            if high_severity:
                for c in high_severity:
                    logger.warning(f"[批判] 高严重度问题: {c.get('issue')} → {c.get('suggestion')}")
            return result

        logger.info("[批判] 批判未产出有效结果，继续当前计划")
        return None

    def _apply_criticism_to_plan(
        self, plan: Dict[str, Any], criticism: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将批判建议应用到计划中。

        Args:
            plan: 当前计划
            criticism: 批判结果

        Returns:
            修正后的计划
        """
        revised = plan.copy()
        revision = criticism.get("plan_revision", {})

        # 修正主要假设
        if revision.get("hypothesis_revised"):
            revised["primary_hypothesis"] = revision["hypothesis_revised"]
            logger.info(f"[规划修正] 假设修订: {revision['hypothesis_revised']}")

        # 修正优先行动
        if revision.get("priority_actions_revised"):
            revised["strategy"] = revised.get("strategy", {})
            revised["strategy"]["priority_actions"] = revision["priority_actions_revised"]
            logger.info("[规划修正] 优先行动已调整")

        # 阶段转换
        phase_change = revision.get("phase_change")
        if phase_change and phase_change.get("to"):
            new_phase = phase_change["to"]
            if new_phase in [p.value for p in Phase]:
                self.current_phase = Phase(new_phase)
                revised["strategy"]["current_phase"] = new_phase
                logger.info(f"[规划修正] 阶段转换: {phase_change.get('from')} → {new_phase}, 原因: {phase_change.get('reason')}")

        # 新增风险
        new_risks = revision.get("new_risks", [])
        if new_risks:
            risk_assessment = revised.get("risk_assessment", {})
            existing_urgent = risk_assessment.get("urgent_findings", [])
            risk_assessment["urgent_findings"] = existing_urgent + new_risks
            revised["risk_assessment"] = risk_assessment

        # 被忽略的信息
        missing_info = revision.get("missing_info", [])
        if missing_info:
            strategy = revised.get("strategy", {})
            existing_gaps = strategy.get("info_gaps", [])
            strategy["info_gaps"] = existing_gaps + missing_info
            revised["strategy"] = strategy

        return revised

    def decide_next_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """根据规划决策下一步具体操作（tool call）。

        这是规划器到执行器的桥梁：将宏观策略转化为具体的 tool call。

        Args:
            plan: 当前规划结果

        Returns:
            行动决策字典，包含：
            - action: ActionType 枚举值
            - target: 行动目标（如具体问题、检查项目等）
            - reason: 行动理由
            - phase: 当前阶段
        """
        strategy = plan.get("strategy", {})
        priority_actions = strategy.get("priority_actions", [])

        # 安全检查：总行动数上限
        if len(self.action_history) >= self.max_total_actions:
            logger.warning("[规划] 达到行动上限，强制进入诊断阶段")
            return {
                "action": ActionType.PRESCRIBE_TREATMENT,
                "target": "基于已有信息提交诊疗方案",
                "reason": "达到最大行动次数，必须提交方案",
                "phase": self.current_phase.value,
            }

        # 从优先行动列表中提取第一个行动
        if priority_actions:
            first = priority_actions[0]
            action_str = first.get("action", "")
            target = first.get("target", "")
            reason = first.get("reason", "")

            # 映射到 ActionType
            action_map = {
                "ask": ActionType.ASK_PATIENT,
                "examine": ActionType.ORDER_EXAMINATION,
                "diagnose": ActionType.PRESCRIBE_TREATMENT,
                "treat": ActionType.PRESCRIBE_TREATMENT,
            }
            action = action_map.get(action_str, ActionType.ASK_PATIENT)

            # 阶段一致性校验
            action = self._validate_phase_consistency(action)

            return {
                "action": action,
                "target": target,
                "reason": reason,
                "phase": self.current_phase.value,
            }

        # 回退：根据当前阶段决定默认行动
        return self._default_action_for_phase()

    def _validate_phase_consistency(self, action: ActionType) -> ActionType:
        """校验行动与当前阶段的一致性，必要时调整阶段。

        Args:
            action: 计划的行动

        Returns:
            校验后的行动
        """
        # 问诊轮次上限检查
        if action == ActionType.ASK_PATIENT and self.inquiry_rounds >= self.max_inquiry_rounds:
            logger.info(f"[规划] 问诊已达上限({self.max_inquiry_rounds}轮)，转入检查阶段")
            self.current_phase = Phase.EXAMINATION
            return ActionType.ORDER_EXAMINATION

        # 检查轮次上限检查
        if action == ActionType.ORDER_EXAMINATION and self.exam_rounds >= self.max_exam_rounds:
            logger.info(f"[规划] 检查已达上限({self.max_exam_rounds}轮)，转入诊断阶段")
            self.current_phase = Phase.DIAGNOSIS
            return ActionType.PRESCRIBE_TREATMENT

        # 阶段转换：问诊→检查
        if self.current_phase == Phase.INQUIRY and action == ActionType.ORDER_EXAMINATION:
            self.current_phase = Phase.EXAMINATION

        # 阶段转换：检查→诊断/治疗
        if self.current_phase == Phase.EXAMINATION and action == ActionType.PRESCRIBE_TREATMENT:
            self.current_phase = Phase.DIAGNOSIS

        return action

    def _default_action_for_phase(self) -> Dict[str, Any]:
        """根据当前阶段返回默认行动。"""
        defaults = {
            Phase.INITIAL: (ActionType.ASK_PATIENT, "收集主诉和基本病史"),
            Phase.INQUIRY: (ActionType.ASK_PATIENT, "继续追问细节"),
            Phase.EXAMINATION: (ActionType.ORDER_EXAMINATION, "申请检查项目"),
            Phase.DIAGNOSIS: (ActionType.PRESCRIBE_TREATMENT, "提交诊疗方案"),
            Phase.TREATMENT: (ActionType.PRESCRIBE_TREATMENT, "提交诊疗方案"),
            Phase.COMPLETED: (ActionType.PRESCRIBE_TREATMENT, "诊疗已完成"),
        }
        action, target = defaults.get(self.current_phase, (ActionType.PRESCRIBE_TREATMENT, "提交方案"))
        return {
            "action": action,
            "target": target,
            "reason": f"阶段 {self.current_phase.value} 的默认行动",
            "phase": self.current_phase.value,
        }

    def _fallback_plan(
        self, collected_info: Dict[str, Any], exam_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """规划失败时的回退策略。"""
        symptoms = collected_info.get("symptoms", [])
        has_chief_complaint = bool(collected_info.get("chief_complaint"))
        has_exam = bool(exam_results)

        if not has_chief_complaint:
            phase = "inquiry"
            priority = [{"action": "ask", "target": "主诉和现病史", "reason": "尚未收集基本信息"}]
        elif not has_exam:
            phase = "examination"
            priority = [{"action": "examine", "target": "基础检查", "reason": "问诊已有基础信息，需检查支持诊断"}]
        else:
            phase = "diagnosis"
            priority = [{"action": "diagnose", "target": "综合诊断", "reason": "信息已较充分，可以诊断"}]

        return {
            "primary_hypothesis": "待定",
            "hypothesis_confidence": 0.3,
            "differential_diagnoses": [],
            "strategy": {
                "current_phase": phase,
                "phase_goal": "完成诊疗",
                "priority_actions": priority,
                "info_gaps": [],
                "decision_points": [],
            },
            "phase_plan": {},
            "risk_assessment": {"urgent_findings": [], "red_flags": [], "safety_constraints": []},
        }

    def should_replan(self, new_info: Dict[str, Any] = None) -> bool:
        """判断是否需要重新规划。

        触发条件：
        1. 批判评估为 needs_replan
        2. 发现新的高风险信息
        3. 阶段转换时
        4. 关键假设被推翻

        Args:
            new_info: 新获取的信息

        Returns:
            是否需要重新规划
        """
        # 批判建议重规划
        if self.criticism_result and self.criticism_result.get("overall_assessment") == "needs_replan":
            return True

        # 发现高风险信息
        if new_info:
            red_flags = new_info.get("red_flags", [])
            if red_flags:
                logger.info(f"[规划] 发现危险信号: {red_flags}，触发重规划")
                return True

        return False

    def get_plan_summary(self) -> str:
        """获取当前规划的摘要文本，用于注入后续 prompt。"""
        if not self.current_plan:
            return "暂无诊疗规划。"

        lines = ["【当前诊疗规划】"]
        lines.append(f"主要假设: {self.current_plan.get('primary_hypothesis', '未知')}")
        lines.append(f"置信度: {self.current_plan.get('hypothesis_confidence', '?')}")

        strategy = self.current_plan.get("strategy", {})
        lines.append(f"当前阶段: {strategy.get('current_phase', '?')}")
        lines.append(f"阶段目标: {strategy.get('phase_goal', '?')}")

        dd = self.current_plan.get("differential_diagnoses", [])
        if dd:
            lines.append("鉴别诊断:")
            for i, d in enumerate(dd[:3], 1):
                lines.append(f"  {i}. {d.get('diagnosis', '?')} (可能性: {d.get('likelihood', '?')})")

        gaps = strategy.get("info_gaps", [])
        if gaps:
            lines.append(f"信息缺口: {', '.join(gaps[:5])}")

        risk = self.current_plan.get("risk_assessment", {})
        red_flags = risk.get("red_flags", [])
        if red_flags:
            lines.append(f"危险信号: {', '.join(red_flags)}")

        return "\n".join(lines)


class MyDoctorAgent(BaseDoctorAgent):
    """参赛医生 Agent 实现。

    采用规划-执行-反思架构：
    - Planner 作为中枢，制定全局策略并通过 Reflection/Criticism 自我修正
    - LLM 进行深度推理，决策下一步具体操作（tool call）
    - Memory 整合历史经验，指导规划决策
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prompt = DoctorPrompt()
        self.memory = DoctorMemory(config)
        self.llm = LLMClient(config)
        self.max_ask_rounds = config.get("max_ask_rounds", 5)
        self.max_exam_rounds = config.get("max_exam_rounds", 3)
        self.log_llm_prompts = config.get("log_llm_prompts", False)
        execution_config = config.get("execution", {}) or {}
        self.fast_mode = bool(execution_config.get("fast_mode", False))
        self.case_timeout_seconds = float(execution_config.get("case_timeout_seconds", 0) or 0)
        self.fallback_reserve_seconds = float(
            execution_config.get("fallback_reserve_seconds", 20) or 20
        )
        self.train_post_submit_reserve_seconds = float(
            execution_config.get("train_post_submit_reserve_seconds", 40) or 40
        )
        self.train_evaluation_timeout_seconds = float(
            execution_config.get("train_evaluation_timeout_seconds", 12) or 12
        )
        self.train_reflection_timeout_seconds = float(
            execution_config.get("train_reflection_timeout_seconds", 24) or 24
        )
        self.max_llm_calls_per_case = int(execution_config.get("max_llm_calls_per_case", 0) or 0)
        self.skip_train_reflection = bool(
            execution_config.get("skip_train_reflection", self.fast_mode)
        )
        self.planner_criticism_max_calls = int(
            execution_config.get("planner_criticism_max_calls", 0 if self.fast_mode else 2)
        )
        self.planner_max_total_actions = int(
            execution_config.get("planner_max_total_actions", 6 if self.fast_mode else 15)
        )
        self.fast_initial_question = execution_config.get(
            "fast_initial_question",
            "请描述这次最主要的不适、开始时间、伴随症状、既往病史、用药史和过敏史。",
        )
        self.fast_exam_items = execution_config.get("fast_exam_items") or [
            "体格检查",
            "超声心动图",
            "心电图（ECG）",
            "胸部X线检查（CXR）",
            "心导管检查",
            "血常规",
            "C反应蛋白",
            "胸部CT",
            "尿常规",
            "腹部B超",
            "甲状腺功能",
        ]
        self.fast_max_exam_items = int(execution_config.get("fast_max_exam_items", 10) or 10)

        learning_config = config.get("learning", {}) or {}
        self.freeze_active_learning = bool(
            learning_config.get("freeze_active_knowledge", True)
        )

        # 静态医学知识库（症状倒排 + 检查规范化 + RAG）
        ref_dir = config.get("ref_data_dir", "data/ref_data")
        self.knowledge = KnowledgeBase(
            ref_dir=ref_dir,
            allow_auto_alias_promotion=bool(
                learning_config.get("auto_promote_exam_aliases", False)
            )
            and not self.freeze_active_learning,
        )
        self.diagnosis_chain_enabled = bool(
            (config.get("diagnosis", {}) or {}).get("enabled", True)
        )
        self.legacy_candidate_submission = bool(
            (config.get("diagnosis", {}) or {}).get("legacy_candidate_submission", True)
        )
        self.clinical_normalizer = ClinicalEvidenceNormalizer(ref_dir=ref_dir)
        self.evidence_agent = EvidenceAgent(ref_dir=ref_dir, normalizer=self.clinical_normalizer)
        self.diagnosis_engine = DiagnosisDecisionEngine(config=config, ref_dir=ref_dir)
        self.exam_agent = ExamStrategyAgent(self.knowledge)
        self.inquiry_agent = InquiryStrategyAgent(self.knowledge)
        self.quality_agent = QualityAgent(
            self.knowledge,
            allowed_diagnoses=self.diagnosis_engine.knowledge.allowed_names,
        )
        self.treatment_agent = TreatmentStrategyAgent(
            self.knowledge,
            diagnostic_knowledge=self.diagnosis_engine.knowledge,
        )
        self.treatment_safety = TreatmentSafetyGate(self.diagnosis_engine.knowledge)
        self.diagnosis_critic = DiagnosisCritic(
            config=config,
            knowledge=self.diagnosis_engine.knowledge,
            resolver=self.diagnosis_engine.resolver,
            llm_chat_json=self._llm_chat_json,
        )
        self.structural_agent = StructuralDiagnosisAgent()
        self.evidence_engine = EvidenceDiagnosisEngine(ref_dir=ref_dir)
        diagnosis_config = config.get("diagnosis", {}) or {}
        self.diagnostic_learning = DiagnosticLearningStore(
            path=diagnosis_config.get(
                "learning_path",
                os.path.join(ref_dir, "pending_diagnostic_rules.json"),
            )
        )
        self._case_started_at = 0.0
        self._case_deadline = 0.0
        self._case_clinical_deadline = 0.0
        self._case_post_submit_reserve_seconds = 0.0
        self._last_diagnosis_audit: Dict[str, Any] = {}

        # 规划器（延迟初始化，因为需要绑定异步方法）
        self._planner: Optional[Planner] = None

        # LLM 成本可观测：调用次数统计
        self._llm_call_count = 0
        self._llm_call_by_kind: Dict[str, int] = {}

        # 相关经验缓存：以 symptoms 元组为 key，避免每次 executor 重查
        self._exp_cache: Dict[tuple, List[Dict[str, Any]]] = {}

        # 自迭代闭环：缺陷检测器 + 策略补丁库（默认启用；可通过 config 关闭）
        self.self_improve_enabled = bool(config.get("self_improve_enabled", True))
        # P1 LLM 归因通道开关（默认开；关闭则 detect_all 退化为纯规则通道）
        self.self_improve_use_llm_attribute = bool(
            config.get("self_improve_use_llm_attribute", True)
        )
        self.detector: Optional[DefectDetector] = None
        self.policy_store: Optional[PolicyStore] = None
        if self.self_improve_enabled:
            try:
                # 注入 llm_chat 以启用 LLM 归因通道；critic 内部会在 use_llm=False 或 llm_chat=None 时自动退化
                self.detector = DefectDetector(
                    knowledge=self.knowledge,
                    llm_chat=self._llm_chat if self.self_improve_use_llm_attribute else None,
                )
                policy_path = config.get(
                    "policy_store_path", "data/memory_data/policies.json"
                )
                self.policy_store = PolicyStore(store_path=policy_path)
                sanitation = self.policy_store.sanitize_shadow_patches()
                logger.info(
                    f"[自迭代] 已启用 detector + policy_store "
                    f"(现有补丁 {len(self.policy_store.patches)} 项, "
                    f"清退不可执行 shadow {sanitation['retired']} 项)"
                )
            except Exception as _e:
                logger.warning(f"[自迭代] 初始化失败，降级为无自迭代模式: {_e}")
                self.detector = None
                self.policy_store = None
        self.memory_manager = DoctorAgentMemory(
            config=config,
            episodic_memory=self.memory,
            semantic_memory=self.knowledge,
            policy_store=self.policy_store,
        )
        logger.info(
            "[MemorySystem] 已启用结构化记忆层: %s",
            json.dumps(self.memory_manager.stats(), ensure_ascii=False),
        )

    def _get_cached_experience(self, collected_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于多维特征获取相关经验，带缓存。

        召回维度: symptoms + chief_complaint + past_history + age_bucket + gender。
        缓存 key 结合症状与年龄段/性别，避免不同患者互相污染。

        Args:
            collected_info: 已收集患者信息

        Returns:
            相关历史经验列表
        """
        symptoms = collected_info.get("symptoms", []) or []
        if not symptoms and not collected_info.get("chief_complaint"):
            return []
        key = (
            tuple(sorted(str(s) for s in symptoms)),
            str(collected_info.get("chief_complaint", ""))[:64],
            self.memory._bucket_age(collected_info.get("age", "")),
            str(collected_info.get("gender", "")),
        )
        if key not in self._exp_cache:
            try:
                # 优先多维召回，失败降级到症状召回
                if hasattr(self.memory, "search_relevant_experience_multi"):
                    self._exp_cache[key] = self.memory_manager.search_episodic(
                        collected_info, top_k=3
                    ) or []
                else:
                    self._exp_cache[key] = self.memory.search_relevant_experience(
                        list(symptoms), top_k=3
                    ) or []
            except Exception as e:
                logger.warning(f"[记忆] 检索相关经验失败: {e}")
                self._exp_cache[key] = []
        return self._exp_cache[key]

    def _bump_llm_counter(self, kind: str = "chat") -> None:
        """LLM 调用计数（用于成本监控）。"""
        self._llm_call_count += 1
        self._llm_call_by_kind[kind] = self._llm_call_by_kind.get(kind, 0) + 1

    def _reset_llm_counter(self) -> None:
        """重置计数器（每个患者独立统计）。"""
        self._llm_call_count = 0
        self._llm_call_by_kind = {}
        self._exp_cache = {}

    def _can_call_llm(self, kind: str) -> bool:
        """Return False when the per-case LLM budget has been exhausted."""
        if self.max_llm_calls_per_case <= 0:
            return True
        if self._llm_call_count < self.max_llm_calls_per_case:
            return True
        logger.warning(
            "[LLM] skip %s call: per-case budget reached (%s)",
            kind,
            self.max_llm_calls_per_case,
        )
        return False

    def _get_planner(self) -> Planner:
        """获取或初始化规划器。"""
        if self._planner is None:
            self._planner = Planner(
                prompt=self.prompt,
                llm_chat_json=self._llm_chat_json,
                llm_chat=self._llm_chat,
                memory=self.memory,
            )
            self._planner.max_inquiry_rounds = self.max_ask_rounds
            self._planner.max_exam_rounds = self.max_exam_rounds
            self._planner.criticism_max_calls = max(0, self.planner_criticism_max_calls)
            self._planner.max_total_actions = max(1, self.planner_max_total_actions)
            # 注入策略补丁库（若 agent 已启用）
            if getattr(self, "policy_store", None) is not None:
                self._planner.policy_store = self.policy_store
        return self._planner

    async def _run_case_pipeline(
        self,
        patient_id: str,
        post_submit_reserve_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Run one case with an optional fast path and hard timeout guard."""
        self._case_started_at = time.monotonic()
        post_submit_reserve_seconds = max(0.0, float(post_submit_reserve_seconds or 0))
        if self.case_timeout_seconds > 0:
            maximum_post_reserve = max(
                0.0,
                self.case_timeout_seconds - self.fallback_reserve_seconds - 1.0,
            )
            post_submit_reserve_seconds = min(
                post_submit_reserve_seconds,
                maximum_post_reserve,
            )
        self._case_post_submit_reserve_seconds = post_submit_reserve_seconds
        self._case_deadline = (
            self._case_started_at + self.case_timeout_seconds
            if self.case_timeout_seconds > 0
            else 0.0
        )
        self._case_clinical_deadline = (
            self._case_deadline - post_submit_reserve_seconds
            if self._case_deadline > 0
            else 0.0
        )
        self._last_diagnosis_audit = {}
        runner = (
            self._execute_fast_path(patient_id)
            if self.fast_mode
            else self._execute_with_planner(patient_id)
        )
        if self.case_timeout_seconds <= 0:
            return await runner
        main_timeout = max(
            1.0,
            self.case_timeout_seconds
            - post_submit_reserve_seconds
            - max(0.0, self.fallback_reserve_seconds),
        )
        try:
            return await asyncio.wait_for(runner, timeout=main_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[CaseTimeout] patient=%s clinical path exceeded %.1fs; "
                "using %.1fs reserve for fallback submission",
                patient_id,
                main_timeout,
                max(0.0, self._remaining_case_seconds()),
            )
            return await self._build_timeout_final_result(
                patient_id,
                "临床诊疗链路超过 "
                f"{self.case_timeout_seconds - post_submit_reserve_seconds:.0f} 秒预算，"
                "已触发保底提交。",
            )

    def _remaining_case_seconds(self) -> float:
        deadline = self._case_clinical_deadline or self._case_deadline
        if deadline <= 0:
            return 10_000.0
        return max(0.0, deadline - time.monotonic())

    def _remaining_total_case_seconds(self) -> float:
        if self._case_deadline <= 0:
            return 10_000.0
        return max(0.0, self._case_deadline - time.monotonic())

    async def _build_timeout_final_result(self, patient_id: str, reason: str) -> Dict[str, Any]:
        """Submit a safe final result when the main execution path is interrupted."""
        collected_info = getattr(self, "_last_collected_info", {}) or {}
        exam_results = getattr(self, "_last_exam_results", {}) or {}
        planner = self._get_planner()
        conversation_rounds = len(
            [a for a in planner.action_history if a.get("type") == "ask_patient"]
        )
        fallback = self.quality_agent.default_final_result(reason)
        if self.diagnosis_chain_enabled:
            evidence_graph = self.evidence_agent.build_graph(collected_info, exam_results)
            evidence = evidence_graph.bundle
            decision = self.diagnosis_engine.decide(fallback, [], evidence)
            fallback = self.diagnosis_engine.apply_to_result(fallback, decision, evidence)
        else:
            fallback = self.evidence_engine.review(
                fallback,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            if not fallback.get("_trusted_diagnoses"):
                fallback = self.structural_agent.review(
                    fallback,
                    collected_info=collected_info,
                    exam_results=exam_results,
                )
        fallback = self.quality_agent.review_final_result(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        fallback = self.treatment_agent.review(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        fallback = self.treatment_safety.review(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        try:
            remaining = self._remaining_case_seconds()
            if remaining <= 0.5:
                raise asyncio.TimeoutError("no fallback submission budget remaining")
            submit_result = await asyncio.wait_for(
                self.actions.prescribe_treatment(
                    patient_id=patient_id,
                    diagnosis=fallback.get("diagnosis", []),
                    treatment_plan=fallback.get("treatment_plan", ""),
                    reasoning=fallback.get("reasoning", ""),
                ),
                timeout=max(0.5, remaining - 0.25),
            )
            if isinstance(submit_result, dict):
                fallback.update({k: v for k, v in submit_result.items() if v not in (None, "")})
        except (Exception, asyncio.TimeoutError) as exc:
            logger.warning("[CaseTimeout] fallback submit failed for %s: %s", patient_id, exc)
            fallback.update(
                {
                    "patient_id": patient_id,
                    "caseId": patient_id,
                    "ordered_examinations": list(exam_results.keys()),
                    "finished": True,
                }
            )
        reviewed = self.quality_agent.review_final_result(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        reviewed["_case_elapsed_seconds"] = round(
            max(0.0, time.monotonic() - self._case_started_at), 3
        )
        reviewed["_case_timed_out"] = True
        if self._last_diagnosis_audit:
            self._last_diagnosis_audit["elapsed_seconds"] = reviewed[
                "_case_elapsed_seconds"
            ]
            self._last_diagnosis_audit["timed_out"] = True
        return reviewed

    async def _execute_fast_path(self, patient_id: str) -> Dict[str, Any]:
        """A bounded case path for local training/testing speed and reliability."""
        planner = self._get_planner()
        working_memory = self.memory_manager.start_case(patient_id)
        chat_history: List[Dict[str, str]] = working_memory.chat_history
        collected_info: Dict[str, Any] = working_memory.collected_info
        exam_results: Dict[str, Any] = working_memory.exam_results
        self._last_collected_info = {}
        self._last_exam_results = {}

        logger.info("[FastPath] start patient=%s", patient_id)

        question = str(self.fast_initial_question)
        try:
            answer = await self.actions.ask_patient(
                patient_id=patient_id,
                input_data={"question": question, "chat_history": chat_history},
            )
        except Exception as exc:
            logger.warning("[FastPath] ask_patient failed for %s: %s", patient_id, exc)
            answer = ""

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": str(answer)})
        planner.current_phase = Phase.INQUIRY
        planner._record_action("ask_patient", "fast_initial_inquiry", str(answer)[:120])
        planner.inquiry_rounds = 1

        collected_info = self._fallback_parse_patient_response(str(answer), collected_info)
        if answer and not collected_info.get("chief_complaint"):
            collected_info["chief_complaint"] = str(answer)[:180]
        self.memory_manager.update_collected_info(patient_id, collected_info)
        self._last_collected_info = dict(collected_info)

        symptoms = collected_info.get("symptoms", []) or []
        disease_hits = self.knowledge.recall_diseases_by_symptoms(symptoms, top_k=5) if symptoms else []
        candidate_names = [item.get("name") for item in disease_hits if item.get("name")]
        self.memory_manager.update_candidates(patient_id, candidate_names)

        proposed_items = list(self._fallback_generate_examination_items(collected_info))
        for item in self.fast_exam_items:
            if item and item not in proposed_items:
                proposed_items.append(item)
        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=candidate_names,
            proposed_items=proposed_items,
            existing_results=exam_results,
        )
        normalized_fast_items, _ = self.knowledge.normalize_examinations(self.fast_exam_items)
        raw_exam_items = []
        for item in list(self.fast_exam_items) + normalized_fast_items + strategy.get("items", []):
            if item and item not in raw_exam_items:
                raw_exam_items.append(item)
        exam_items = self.exam_agent.prepare_order_items(
            raw_exam_items,
            collected_info=collected_info,
            candidate_diseases=candidate_names,
            existing_results=exam_results,
            max_items=self.fast_max_exam_items,
        )

        if exam_items:
            try:
                response = await self.actions.order_examination(
                    patient_id=patient_id,
                    items=exam_items,
                    reason="快速路径：基于主诉、症状召回和疾病画像补齐关键检查。",
                )
                new_results = {}
                if response and "results" in response:
                    for exam_name, exam_data in response["results"].items():
                        if isinstance(exam_data, dict) and exam_data.get("status") != "invalid":
                            new_results[exam_name] = exam_data
                if new_results:
                    exam_results.update(new_results)
                    self.memory_manager.update_exam_results(patient_id, new_results)
                    planner._record_action(
                        "order_examination",
                        ",".join(new_results.keys()),
                        "fast_exam_batch",
                    )
                    planner.exam_rounds += 1
            except Exception as exc:
                logger.warning("[FastPath] order_examination failed for %s: %s", patient_id, exc)
        self._last_exam_results = dict(exam_results)

        planner.current_phase = Phase.TREATMENT
        final_result = await self._prescribe(
            patient_id,
            collected_info,
            exam_results,
            chat_history,
            self._get_cached_experience(collected_info),
        )
        planner._record_action("prescribe_treatment", "fast_final_submit", "")
        planner.current_phase = Phase.COMPLETED
        self._last_collected_info = dict(collected_info)
        self._last_exam_results = dict(exam_results)
        self.memory_manager.finish_case(patient_id)
        return final_result

    # ============ 规划驱动执行循环 ============

    async def _execute_with_planner(self, patient_id: str) -> Dict[str, Any]:
        """规划器驱动的核心执行循环。

        替代旧的硬编码流水线（初始问诊→追问→检查→诊断），
        由 Planner 作为中枢，通过 LLM 深度推理决策每一步操作。

        流程：
        1. 初始化状态 → Planner 制定全局策略
        2. 循环：decide_next_action → 执行 → 记录 → 检查重规划
        3. 提交诊疗方案 → 返回结果

        Args:
            patient_id: 患者 ID

        Returns:
            最终诊疗结果
        """
        planner = self._get_planner()
        working_memory = self.memory_manager.start_case(patient_id)
        chat_history: List[Dict[str, str]] = working_memory.chat_history
        collected_info: Dict[str, Any] = working_memory.collected_info
        exam_results: Dict[str, Any] = working_memory.exam_results
        relevant_experience: List[Dict[str, Any]] = []

        logger.info(f"[规划执行] 开始规划驱动诊疗，患者: {patient_id}")

        def _sync_plan(plan_obj: Dict[str, Any]) -> None:
            if isinstance(plan_obj, dict):
                self.memory_manager.update_candidates(
                    patient_id,
                    plan_obj.get("differential_diagnoses", []),
                )

        # 保存执行状态，供后续反思使用
        self._last_collected_info: Dict[str, Any] = {}
        self._last_exam_results: Dict[str, Any] = {}

        # ---- 阶段1：初始问诊（至少收集主诉） ----
        planner.current_phase = Phase.INQUIRY
        collected_info = await self._initial_inquiry(
            patient_id, chat_history, relevant_experience
        )
        self.memory_manager.update_collected_info(patient_id, collected_info)
        self._last_collected_info = dict(collected_info)
        planner._record_action("ask_patient", "初始问诊-主诉", str(collected_info.get("chief_complaint", ""))[:100])
        planner.inquiry_rounds = 1

        # ---- 阶段2：制定全局策略 ----
        plan = await planner.plan(
            collected_info=collected_info,
            exam_results=exam_results,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
        )
        _sync_plan(plan)
        logger.info(f"[规划执行] 初始规划完成，阶段: {planner.current_phase.value}")

        # ---- 阶段3：规划驱动执行循环 ----
        for step in range(planner.max_total_actions):
            # 3.1 决策下一步操作
            action_decision = planner.decide_next_action(plan)
            action_type = action_decision["action"]
            target = action_decision["target"]
            reason = action_decision.get("reason", "")

            logger.info(
                f"[规划执行] 步骤{step + 1}: action={action_type.value}, "
                f"target={target}, phase={planner.current_phase.value}"
            )

            # 3.2 执行操作
            if action_type == ActionType.ASK_PATIENT:
                # 问诊：生成追问问题并询问患者
                result_info = await self._execute_ask_patient(
                    patient_id, chat_history, collected_info,
                    relevant_experience, target, reason,
                )
                if result_info:
                    collected_info = result_info
                    self.memory_manager.update_collected_info(patient_id, collected_info)
                    self._last_collected_info = dict(collected_info)
                    planner._record_action("ask_patient", target, reason)
                    planner.inquiry_rounds += 1
                else:
                    # 问诊被跳过（信息已足够或LLM判断无需追问），触发重规划
                    planner._record_action("ask_patient_skipped", target, "信息已足够，跳过追问")
                    logger.info("[规划执行] 问诊被跳过，触发重规划以转换阶段")
                    plan = await planner.plan(
                        collected_info=collected_info,
                        exam_results=exam_results,
                        chat_history=chat_history,
                        relevant_experience=relevant_experience,
                    )
                    _sync_plan(plan)
                    continue

            elif action_type == ActionType.ORDER_EXAMINATION:
                # 检查：申请检查项目
                result_exams = await self._execute_order_examination(
                    patient_id, collected_info, exam_results,
                    relevant_experience, target, reason,
                )
                if result_exams:
                    exam_results.update(result_exams)
                    self.memory_manager.update_exam_results(patient_id, result_exams)
                    self._last_exam_results = dict(exam_results)
                    planner._record_action("order_examination", target, reason)
                    planner.exam_rounds += 1
                else:
                    # 检查被跳过（检查已足够），触发重规划
                    planner._record_action("examination_skipped", target, "检查已足够，跳过")
                    logger.info("[规划执行] 检查被跳过，触发重规划以转换阶段")
                    plan = await planner.plan(
                        collected_info=collected_info,
                        exam_results=exam_results,
                        chat_history=chat_history,
                        relevant_experience=relevant_experience,
                    )
                    _sync_plan(plan)
                    continue

            elif action_type == ActionType.PRESCRIBE_TREATMENT:
                # 诊断/治疗：提交诊疗方案
                final_result = await self._prescribe(
                    patient_id, collected_info, exam_results,
                    chat_history, relevant_experience,
                )
                planner._record_action("prescribe_treatment", "提交诊疗方案", "")
                planner.current_phase = Phase.COMPLETED
                logger.info(f"[规划执行] 诊疗方案已提交，完成")
                # 保存执行状态供反思使用
                self._last_collected_info = collected_info
                self._last_exam_results = exam_results
                self.memory_manager.finish_case(patient_id)
                return final_result

            elif action_type == ActionType.REPLAN:
                # 重规划
                logger.info(f"[规划执行] 触发重规划")
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)
                continue

            # 3.3 检查是否需要重规划
            if planner.should_replan({"red_flags": collected_info.get("red_flags", [])}):
                logger.info(f"[规划执行] 检测到重规划需求")
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)
                continue

            # 3.4 阶段转换时更新策略（非每步都重规划，节省 LLM 调用）
            # 使用 Phase 枚举直接比较，避免字符串误比对
            plan_phase_str = plan.get("strategy", {}).get("current_phase", "")
            try:
                plan_phase = Phase(plan_phase_str) if plan_phase_str else planner.current_phase
            except ValueError:
                plan_phase = planner.current_phase
            if planner.current_phase != plan_phase:
                logger.info(
                    f"[规划执行] 阶段转换 {plan_phase.value} → {planner.current_phase.value}，更新策略"
                )
                relevant_experience = self._get_cached_experience(collected_info)
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)

            # 3.5 安全检查：如果循环结束仍未提交方案，强制提交
            if step == planner.max_total_actions - 1:
                logger.warning("[规划执行] 达到最大步骤数，强制提交诊疗方案")
                final_result = await self._prescribe(
                    patient_id, collected_info, exam_results,
                    chat_history, relevant_experience,
                )
                planner.current_phase = Phase.COMPLETED
                self._last_collected_info = collected_info
                self._last_exam_results = exam_results
                self.memory_manager.finish_case(patient_id)
                return final_result

        # 兜底返回
        self._last_collected_info = collected_info
        self._last_exam_results = exam_results
        final_result = await self._prescribe(
            patient_id, collected_info, exam_results,
            chat_history, relevant_experience,
        )
        self.memory_manager.finish_case(patient_id)
        return final_result

    async def _execute_ask_patient(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]],
        target: str,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """执行单次问诊操作。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            collected_info: 已收集信息
            relevant_experience: 相关经验
            target: 问诊目标（来自规划器）
            reason: 问诊理由（来自规划器）

        Returns:
            更新后的患者信息，失败返回 None
        """
        # 基于症状检索相关经验（带缓存）
        cached_exp = self._get_cached_experience(collected_info)
        if cached_exp:
            relevant_experience = cached_exp

        # 思考：生成鉴别诊断，判断信息是否足够
        thinking = await self._think(
            collected_info, {}, chat_history, "inquiry", relevant_experience
        )
        _cands = None
        if thinking and isinstance(thinking, dict):
            _cands = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
        inquiry_strategy = self.inquiry_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
        )

        # 判断信息是否已足够
        if thinking and "is_sufficient" in thinking:
            if thinking.get("is_sufficient"):
                logger.info("[问诊] 思考判断信息已足够，跳过追问")
                return None

        # 构建追问 prompt（注入思考结果和规划目标）
        follow_up_prompt = self.prompt.build_follow_up_prompt(
            collected_info=collected_info,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
            thinking=thinking,
        )
        messages = [
            {"role": "system", "content": follow_up_prompt},
            {
                "role": "user",
                "content": (
                    f"请针对以下目标生成追问问题：{target}。理由：{reason}。"
                    f"优先覆盖这些关键追问：{inquiry_strategy.get('questions', [])}。"
                    f"注意排查红旗信号：{inquiry_strategy.get('red_flags', [])}。"
                    "如果信息已足够，请返回空字符串。"
                ),
            },
        ]

        question = await self._llm_chat(messages, temperature=0.5)
        if not question or question.strip() == "":
            logger.info("[问诊] LLM 判断无需追问")
            return None

        logger.info(f"[问诊] 追问: {question}")

        answer = await self.actions.ask_patient(
            patient_id=patient_id,
            input_data={"question": question, "chat_history": chat_history},
        )

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": answer})

        # 提取结构化信息
        updated_info = await self._extract_patient_info(answer, collected_info)
        logger.info(f"[问诊] 追问回复: {answer[:200]}...")

        return updated_info

    async def _execute_order_examination(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]],
        target: str,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """执行单次检查操作。

        Args:
            patient_id: 患者 ID
            collected_info: 已收集信息
            exam_results: 已有检查结果
            relevant_experience: 相关经验
            target: 检查目标（来自规划器）
            reason: 检查理由（来自规划器）

        Returns:
            新增检查结果字典，失败返回 None
        """
        # 基于症状检索相关经验（带缓存）
        cached_exp = self._get_cached_experience(collected_info)
        if cached_exp:
            relevant_experience = cached_exp

        # 思考：更新鉴别诊断，判断检查是否足够
        thinking = await self._think(
            collected_info, exam_results, [], "examination", relevant_experience
        )

        # 判断检查是否已足够
        if exam_results and thinking and "is_sufficient" in thinking:
            if thinking.get("is_sufficient"):
                logger.info("[检查] 思考判断检查已足够，跳过")
                return None

        # 从 knowledge 构建 RAG 上下文（症状 + 候选疾病）
        _sym = collected_info.get("symptoms") or []
        _cands = None
        if thinking and isinstance(thinking, dict):
            _cands = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
        try:
            knowledge_context = self.knowledge.build_rag_context(_sym, _cands)
        except Exception:
            knowledge_context = ""
        try:
            if getattr(self, "memory_manager", None):
                memory_context = self.memory_manager.build_semantic_context(
                    collected_info, _cands
                )
                if memory_context:
                    knowledge_context = memory_context
        except Exception:
            pass

        # 构建检查申请 prompt（注入思考结果和规划目标）
        exam_prompt = self.prompt.build_examination_prompt(
            collected_info=collected_info,
            exam_results=exam_results,
            relevant_experience=relevant_experience,
            thinking=thinking,
            knowledge_context=knowledge_context,
        )
        messages = [
            {"role": "system", "content": exam_prompt},
            {"role": "user", "content": f"请针对以下目标申请检查：{target}。理由：{reason}。输出 JSON 数组格式的检查项目列表。"},
        ]

        exam_items = await self._llm_generate_examination_items(messages, collected_info)
        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
            proposed_items=exam_items,
            existing_results=exam_results,
        )
        if strategy.get("strong_verification_items"):
            logger.info(f"[检查策略] 强验证检查: {strategy['strong_verification_items']}")
        if strategy.get("red_flag_items"):
            logger.info(f"[检查策略] 红旗补查检查: {strategy['red_flag_items']}")
        if strategy.get("evidence_driven_items"):
            logger.info(f"[检查策略] 证据驱动补查检查: {strategy['evidence_driven_items']}")
        if strategy.get("added_required"):
            logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
        if strategy.get("invalid_items"):
            logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
        exam_items = self.exam_agent.prepare_order_items(
            strategy.get("items", []),
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
            existing_results=exam_results,
            max_items=self.exam_agent.max_new_items,
            add_strong_verification=False,
        )
        if not exam_items:
            logger.info("[检查] 无检查项目需要申请")
            return None

        # 去重
        new_items = [item for item in exam_items if item not in exam_results]
        if not new_items:
            logger.info("[检查] 所有推荐检查已完成")
            return None

        logger.info(f"[检查] 申请检查: {new_items}")

        # 申请检查
        response = await self.actions.order_examination(
            patient_id=patient_id,
            items=new_items,
            reason=reason or "基于问诊信息，需要进一步检查以明确诊断。",
        )

        # 合并检查结果
        new_results = {}
        if response and "results" in response:
            for exam_name, exam_data in response["results"].items():
                if exam_data.get("status") != "invalid":
                    new_results[exam_name] = exam_data

        return new_results if new_results else None

    # ============ 训练流程（规划驱动） ============

    async def train(self, patient_id: str) -> Dict[str, Any]:
        """训练流程：使用规划器驱动诊疗，并在结束后评估反思。

        Args:
            patient_id: 患者 ID
        """
        logger.info(f"[Train] 开始训练患者: {patient_id}")
        self._reset_llm_counter()
        # 跨患者软复用：重置流程状态但保留可迁移的教��
        if self._planner is not None:
            self._planner.soft_reset(keep_lessons=True)

        # 规划器驱动诊疗
        final_result = await self._run_case_pipeline(
            patient_id,
            post_submit_reserve_seconds=self.train_post_submit_reserve_seconds,
        )

        # 训练阶段：先独立获取评估，再执行反思。反思失败不能伪装成评估失败。
        report: Dict[str, Any] = {}
        evaluation_error = ""
        reflection_error = ""
        try:
            evaluation_call = self.actions.evaluation(
                patient_id=patient_id, final_result=final_result
            )
            if self.case_timeout_seconds > 0:
                available = (
                    self._remaining_total_case_seconds()
                    - self.train_reflection_timeout_seconds
                    - 1.0
                )
                evaluation_budget = min(
                    self.train_evaluation_timeout_seconds,
                    max(0.0, available),
                )
                if evaluation_budget < 0.5:
                    evaluation_call.close()
                    raise asyncio.TimeoutError("no evaluation budget remaining")
                raw_report = await asyncio.wait_for(
                    evaluation_call,
                    timeout=evaluation_budget,
                )
            else:
                raw_report = await evaluation_call
            if not isinstance(raw_report, dict):
                raise TypeError("evaluation response must be a JSON object")
            report = raw_report
            logger.info(
                f"[Train] 患者 {patient_id} 评估报告: "
                f"{json.dumps(report, ensure_ascii=False, indent=2)}"
            )
        except asyncio.TimeoutError:
            evaluation_error = (
                f"evaluation exceeded {self.train_evaluation_timeout_seconds:.0f}s budget"
            )
            logger.warning(f"[Train] {evaluation_error}")
        except Exception as exc:
            evaluation_error = str(exc)
            logger.warning(f"[Train] 获取评估报告失败: {exc}")

        if report:
            notes_before_reflection = len(getattr(self.memory, "notes", []) or [])
            try:
                if self.skip_train_reflection:
                    self._save_fast_reflection(
                        patient_id,
                        report,
                        self._last_collected_info,
                        self._last_exam_results,
                    )
                else:
                    reflection_call = self._reflect_and_save(
                        patient_id, report,
                        self._last_collected_info,
                        self._last_exam_results,
                    )
                    if self.case_timeout_seconds > 0:
                        reflection_budget = min(
                            self.train_reflection_timeout_seconds,
                            max(0.0, self._remaining_total_case_seconds() - 1.0),
                        )
                        if reflection_budget < 0.5:
                            reflection_call.close()
                            raise asyncio.TimeoutError("no reflection budget remaining")
                        await asyncio.wait_for(
                            reflection_call,
                            timeout=reflection_budget,
                        )
                    else:
                        await reflection_call
            except asyncio.TimeoutError:
                reflection_error = (
                    f"reflection exceeded {self.train_reflection_timeout_seconds:.0f}s budget"
                )
                logger.warning(f"[Train] {reflection_error}; using deterministic reflection")
                if len(getattr(self.memory, "notes", []) or []) == notes_before_reflection:
                    try:
                        self._save_fast_reflection(
                            patient_id,
                            report,
                            self._last_collected_info,
                            self._last_exam_results,
                        )
                    except Exception as fallback_exc:
                        reflection_error += f"; fallback failed: {fallback_exc}"
            except Exception as exc:
                reflection_error = str(exc)
                logger.warning(f"[Train] 反思或记忆保存失败: {exc}")

        training_elapsed = round(max(0.0, time.monotonic() - self._case_started_at), 3)
        if self._last_diagnosis_audit:
            self._last_diagnosis_audit["training_elapsed_seconds"] = training_elapsed

        # 输出记忆统计
        stats = self.memory.get_statistics()
        logger.info(f"[Train] 记忆统计: {json.dumps(stats, ensure_ascii=False)}")
        logger.info(
            f"[Train] LLM 调用统计: 总{self._llm_call_count}次, "
            f"明细={json.dumps(self._llm_call_by_kind, ensure_ascii=False)}"
        )
        train_result = self._build_training_result(
            patient_id=patient_id,
            final_result=final_result,
            report=report,
            evaluation_error=evaluation_error,
            reflection_error=reflection_error,
        )
        self._last_train_result = train_result
        logger.info(f"[Train] 完成训练患者: {patient_id}")
        return train_result

    def _build_training_result(
        self,
        patient_id: str,
        final_result: Dict[str, Any],
        report: Dict[str, Any],
        evaluation_error: str = "",
        reflection_error: str = "",
    ) -> Dict[str, Any]:
        """Build a compact, secret-free record for batch training reports."""
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}

        def _names(value: Any) -> List[str]:
            if isinstance(value, str):
                value = [value]
            return list(
                dict.fromkeys(
                    str(item).strip() for item in (value or []) if str(item).strip()
                )
            )

        def _metric(*keys: str) -> Optional[float]:
            for key in keys:
                value = report.get(key)
                try:
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        audit = self._last_diagnosis_audit or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") or []
        top_five = [
            str(item.get("diagnosis"))
            for item in candidates[:5]
            if isinstance(item, dict) and item.get("diagnosis")
        ]
        expected = _names(detail.get("expected") or report.get("finalDiagnosis"))
        submitted = _names(
            detail.get("submitted")
            or report.get("diagnosis")
            or final_result.get("diagnosis")
        )
        recall_at_five = all(name in top_five for name in expected) if expected else None
        critic = audit.get("critic") or {}
        elapsed = audit.get(
            "training_elapsed_seconds",
            audit.get("elapsed_seconds", final_result.get("_case_elapsed_seconds")),
        )
        try:
            elapsed = float(elapsed) if elapsed is not None else None
        except (TypeError, ValueError):
            elapsed = None

        public_final = {
            key: final_result.get(key)
            for key in (
                "patient_id",
                "caseId",
                "diagnosis",
                "treatment_plan",
                "reasoning",
                "ordered_examinations",
                "conversation_rounds",
                "finished",
            )
            if key in final_result
        }
        return {
            "patient_id": patient_id,
            "status": "evaluated" if report else "evaluation_failed",
            "final_result": public_final,
            "expected_diagnoses": expected,
            "submitted_diagnoses": submitted,
            "error_types": self._classify_diagnosis_errors(report) if report else [],
            "metrics": {
                "diagnosis_accuracy": _metric("diagnosisAccuracy", "diagnosis_accuracy"),
                "examination_precision": _metric(
                    "examinationPrecision", "examination_precision"
                ),
                "treatment_overall_score": _metric(
                    "treatmentOverallScore", "treatment_overall_score"
                ),
                "treatment_safety": _metric("treatmentSafety", "treatment_safety"),
                "candidate_recall_at_5": recall_at_five,
            },
            "audit": {
                "elapsed_seconds": elapsed,
                "timed_out": bool(
                    audit.get("timed_out", final_result.get("_case_timed_out", False))
                    or "exceeded" in evaluation_error
                    or "exceeded" in reflection_error
                    or (
                        self.case_timeout_seconds > 0
                        and elapsed is not None
                        and elapsed >= self.case_timeout_seconds
                    )
                ),
                "critic_issues": list(critic.get("issues") or []),
                "critic_llm_used": bool(critic.get("llm_used", False)),
                "llm_calls": self._llm_call_count,
                "llm_calls_by_kind": dict(self._llm_call_by_kind),
            },
            "evaluation_error": evaluation_error,
            "reflection_error": reflection_error,
        }

    # ============ 测试流程（规划驱动） ============

    async def test(self, patient_id: str) -> None:
        """测试流程：使用规划器驱动诊疗并提交结果。

        Args:
            patient_id: 患者 ID
        """
        logger.info(f"[Test] 开始测试患者: {patient_id}")
        self._reset_llm_counter()
        # 跨患者软复用：保留 planner 中的经验教训种子
        if self._planner is not None:
            self._planner.soft_reset(keep_lessons=True)

        # 规划器驱动诊疗
        final_result = await self._run_case_pipeline(patient_id)

        # 保存测试结果供 run_test 收集
        planner = self._get_planner()
        _rounds = int(
            final_result.get("conversation_rounds")
            or len([a for a in planner.action_history if a["type"] == "ask_patient"])
        )
        final_result = self.quality_agent.review_final_result(
            final_result,
            collected_info=getattr(self, "_last_collected_info", {}),
            exam_results=getattr(self, "_last_exam_results", {}),
            conversation_rounds=_rounds,
        )
        _dx = final_result.get("diagnosis", [])
        _tp = final_result.get("treatment_plan", "")
        # 关键：final_result / final_results 双字段，兼容评测器的字段名约定
        _final_payload = {
            "patient_id": patient_id,
            "caseId": patient_id,
            "diagnosis": _dx,
            "treatment_plan": _tp,
            "reasoning": final_result.get("reasoning", "") if isinstance(final_result, dict) else "",
            "conversation_rounds": _rounds,
            "ordered_examinations": final_result.get("ordered_examinations", []) if isinstance(final_result, dict) else [],
            "finished": bool(final_result.get("finished", True)) if isinstance(final_result, dict) else True,
        }
        self._last_test_result = {
            "final_result": _final_payload,
            "final_results": [_final_payload],
            "caseId": patient_id,
            "diagnosis": _dx,
            "treatment_plan": _tp,
            "reasoning": _final_payload["reasoning"],
            "conversation_rounds": _rounds,
            "ordered_examinations": _final_payload["ordered_examinations"],
            "finished": _final_payload["finished"],
        }

        logger.info(
            f"[Test] 完成测试患者: {patient_id}, 结果: "
            f"{json.dumps(final_result, ensure_ascii=False)}"
        )
        logger.info(
            f"[Test] LLM 调用统计: 总{self._llm_call_count}次, "
            f"明细={json.dumps(self._llm_call_by_kind, ensure_ascii=False)}"
        )

    # ============ 初始问诊 ============

    async def _initial_inquiry(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """初始问诊：收集主诉和基本病史。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            收集到的患者信息
        """
        # 检索相关经验
        # 初始阶段还没有症状信息，使用空列表
        # 经验将在后续阶段注入

        # 构建初始问诊 prompt
        system_prompt = self.prompt.build_initial_inquiry_prompt(relevant_experience)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请生成一个初始问诊问题，引导患者描述主要症状。"},
        ]

        # 使用 LLM 生成初始问题
        question = await self._llm_chat(messages)
        if not question:
            question = "请描述这次最主要的不适、开始时间和伴随症状。"

        logger.info(f"[问诊] 初始问诊问题: {question}")

        # 询问患者
        answer = await self.actions.ask_patient(
            patient_id=patient_id,
            input_data={
                "question": question,
                "chat_history": chat_history,
            },
        )

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": answer})

        # 使用 LLM 结构化提取患者回复
        collected_info = await self._extract_patient_info(answer, {})
        logger.info(f"[问诊] 初始问诊回复: {answer[:200]}...")
        logger.info(f"[问诊] 提取信息: {json.dumps(collected_info, ensure_ascii=False, indent=2)[:200]}...")

        return collected_info

    # ============ 追问细节 ============

    async def _follow_up_inquiry(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """追问细节：根据初始问诊结果，追问病史和症状细节。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            collected_info: 已收集的信息
            relevant_experience: 相关历史经验

        Returns:
            更新后的患者信息
        """
        # 基于当前症状检索相关经验
        symptoms = collected_info.get("symptoms", [])
        if symptoms:
            relevant_experience = self.memory.search_relevant_experience(symptoms, top_k=3)

        for round_idx in range(self.max_ask_rounds - 1):
            # 思考：生成鉴别诊断，判断信息是否足够
            thinking = await self._think(
                collected_info, {}, chat_history, "inquiry", relevant_experience
            )
            _cands_follow = None
            if thinking and isinstance(thinking, dict):
                _cands_follow = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
            inquiry_strategy = self.inquiry_agent.recommend(
                collected_info=collected_info,
                candidate_diseases=_cands_follow if isinstance(_cands_follow, list) else None,
            )

            # 判断信息是否已足够（从思考结果中获取，回退到旧方法）
            if thinking and "is_sufficient" in thinking:
                is_sufficient = thinking["is_sufficient"]
                if thinking.get("key_unknowns"):
                    logger.info(f"[问诊] 信息缺口: {thinking['key_unknowns']}")
            else:
                is_sufficient = await self._check_info_sufficient(
                    collected_info, round_idx + 2, self.max_ask_rounds
                )

            if is_sufficient:
                logger.info(f"[问诊] 信息已足够，结束问诊")
                break

            # 构建追问 prompt（注入思考结果）
            follow_up_prompt = self.prompt.build_follow_up_prompt(
                collected_info=collected_info,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                thinking=thinking,
            )
            messages = [
                {"role": "system", "content": follow_up_prompt},
                {
                    "role": "user",
                    "content": (
                        "请生成一个追问问题。"
                        f"优先覆盖这些关键追问：{inquiry_strategy.get('questions', [])}。"
                        f"注意排查红旗信号：{inquiry_strategy.get('red_flags', [])}。"
                        "如果信息已足够，请返回空字符串。"
                    ),
                },
            ]

            # 使用 LLM 生成追问问题
            question = await self._llm_chat(messages, temperature=0.5)

            if not question or question.strip() == "":
                logger.info(f"[问诊] LLM 判断无需追问，结束问诊")
                break

            logger.info(f"[问诊] 第{round_idx + 2}轮追问: {question}")

            answer = await self.actions.ask_patient(
                patient_id=patient_id,
                input_data={
                    "question": question,
                    "chat_history": chat_history,
                },
            )

            chat_history.append({"from": "doctor", "text": question})
            chat_history.append({"from": "patient", "text": answer})

            # 使用 LLM 结构化提取追问回复
            collected_info = await self._extract_patient_info(answer, collected_info)

            logger.info(f"[问诊] 第{round_idx + 2}轮追问回复: {answer[:200]}...")

        return collected_info

    # ============ 申请检查 ============

    async def _order_examinations(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据问诊信息申请检查。

        Args:
            patient_id: 患者 ID
            collected_info: 收集到的患者信息
            relevant_experience: 相关历史经验

        Returns:
            检查结果
        """
        # 基于症状检索相关经验
        symptoms = collected_info.get("symptoms", [])
        if symptoms:
            relevant_experience = self.memory.search_relevant_experience(symptoms, top_k=3)

        exam_results = {}

        for exam_round in range(self.max_exam_rounds):
            # 思考：更新鉴别诊断，判断检查是否足够
            thinking = await self._think(
                collected_info, exam_results, [], "examination", relevant_experience
            )

            # 判断检查是否已足够（从思考结果中获取，回退到旧方法）
            if exam_results:
                if thinking and "is_sufficient" in thinking:
                    is_sufficient = thinking["is_sufficient"]
                    if thinking.get("key_unknowns"):
                        logger.info(f"[检查] 建议补充: {thinking['key_unknowns']}")
                else:
                    is_sufficient = await self._check_exam_sufficient(
                        collected_info, exam_results, exam_round + 1, self.max_exam_rounds
                    )
                if is_sufficient:
                    logger.info(f"[检查] 检查已足够，结束检查")
                    break

            # 从 knowledge 构建 RAG 上下文
            _sym2 = collected_info.get("symptoms") or []
            _cands2 = None
            if thinking and isinstance(thinking, dict):
                _cands2 = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
            try:
                knowledge_context2 = self.knowledge.build_rag_context(_sym2, _cands2)
            except Exception:
                knowledge_context2 = ""
            try:
                if getattr(self, "memory_manager", None):
                    memory_context2 = self.memory_manager.build_semantic_context(
                        collected_info, _cands2
                    )
                    if memory_context2:
                        knowledge_context2 = memory_context2
            except Exception:
                pass

            # 构建检查申请 prompt（注入思考结果）
            exam_prompt = self.prompt.build_examination_prompt(
                collected_info=collected_info,
                exam_results=exam_results,
                relevant_experience=relevant_experience,
                thinking=thinking,
                knowledge_context=knowledge_context2,
            )
            messages = [
                {"role": "system", "content": exam_prompt},
                {"role": "user", "content": "请输出需要申请的检查项目列表（JSON 数组格式）。"},
            ]

            # 使用 LLM 生成检查项目
            exam_items = await self._llm_generate_examination_items(messages, collected_info)
            strategy = self.exam_agent.recommend(
                collected_info=collected_info,
                candidate_diseases=_cands2 if isinstance(_cands2, list) else None,
                proposed_items=exam_items,
                existing_results=exam_results,
            )
            if strategy.get("strong_verification_items"):
                logger.info(f"[检查策略] 强验证检查: {strategy['strong_verification_items']}")
            if strategy.get("red_flag_items"):
                logger.info(f"[检查策略] 红旗补查检查: {strategy['red_flag_items']}")
            if strategy.get("evidence_driven_items"):
                logger.info(f"[检查策略] 证据驱动补查检查: {strategy['evidence_driven_items']}")
            if strategy.get("added_required"):
                logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
            if strategy.get("invalid_items"):
                logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
            exam_items = self.exam_agent.prepare_order_items(
                strategy.get("items", []),
                collected_info=collected_info,
                candidate_diseases=_cands2 if isinstance(_cands2, list) else None,
                existing_results=exam_results,
                max_items=self.exam_agent.max_new_items,
                add_strong_verification=False,
            )

            if not exam_items:
                logger.info(f"[检查] 无更多检查需要申请")
                break

            # 去重：排除已检查的项目
            new_items = [item for item in exam_items if item not in exam_results]
            if not new_items:
                logger.info(f"[检查] 所有推荐检查已完成")
                break

            logger.info(f"[检查] 第{exam_round + 1}轮检查: {new_items}")

            # 申请检查
            response = await self.actions.order_examination(
                patient_id=patient_id,
                items=new_items,
                reason=f"基于问诊信息，需要进一步检查以明确诊断。",
            )

            # 合并检查结果
            if response and "results" in response:
                for exam_name, exam_data in response["results"].items():
                    if exam_data.get("status") != "invalid":
                        exam_results[exam_name] = exam_data

        return exam_results

    # ============ 提交诊疗方案 ============

    async def _prescribe(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """综合分析并提交诊疗方案。

        Args:
            patient_id: 患者 ID
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            最终诊疗结果
        """
        conversation_rounds = len([m for m in chat_history if m.get("from") == "doctor"])
        relevant_experience = self._get_cached_experience(collected_info)
        decision = None
        evidence = None

        if self.diagnosis_chain_enabled:
            evidence_graph = self.evidence_agent.build_graph(collected_info, exam_results)
            evidence = evidence_graph.bundle
            planner_candidates = self._planner_candidate_names()
            rag_query = evidence.to_query()
            if planner_candidates:
                rag_query += " 当前鉴别诊断 " + " ".join(planner_candidates)
            rag_chunks = self.memory_manager.search_rag(
                collected_info=collected_info,
                query=rag_query,
                candidate_diseases=planner_candidates or None,
            )
            rag_context = self.memory_manager.render_rag_chunks(rag_chunks)
            preview = self.diagnosis_engine.decide({}, rag_chunks, evidence)
            candidate_table = self.diagnosis_engine.render_candidate_table(preview)
            diagnosis_prompt = self.prompt.build_diagnosis_prompt(
                collected_info=collected_info,
                exam_results=exam_results,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                standard_diseases=self.diagnosis_engine.knowledge.allowed_names,
                rag_context=rag_context,
                evidence_summary=evidence.render_summary(),
                candidate_table=candidate_table,
            )
            messages = [
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content": "请做出诊断并制定治疗方案，以 JSON 格式输出。"},
            ]
            diagnosis_result = await self._llm_generate_diagnosis(messages)
            llm_resolutions = self.diagnosis_engine.resolve_open_candidates(diagnosis_result)
            llm_candidates = []
            for item in llm_resolutions:
                if item.raw_name:
                    llm_candidates.append(item.raw_name)
                if item.canonical_name:
                    llm_candidates.append(item.canonical_name)
            llm_candidates = list(dict.fromkeys(llm_candidates))
            final_candidates = list(
                dict.fromkeys(planner_candidates + llm_candidates)
            )
            final_query = evidence.to_query()
            if final_candidates:
                final_query += " 当前鉴别诊断 " + " ".join(final_candidates)
            final_rag_chunks = self.memory_manager.search_rag(
                collected_info=collected_info,
                query=final_query,
                candidate_diseases=final_candidates or None,
            )
            if final_rag_chunks:
                rag_chunks = final_rag_chunks
            decision = self.diagnosis_engine.decide(diagnosis_result, rag_chunks, evidence)
            critic = await self.diagnosis_critic.review(
                decision,
                evidence,
                remaining_seconds=self._remaining_case_seconds(),
                allow_llm=True,
            )
            self._apply_critic_selection(decision, critic.selected_diagnoses, critic.reason)
            self._restore_legacy_candidate_submission(decision, diagnosis_result, critic)

            evidence_gap_exams = self._recommend_evidence_gap_exams(
                decision=decision,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            recommended_exams = list(
                dict.fromkeys(evidence_gap_exams + list(critic.recommended_exams or []))
            )
            corrective_targets = self._evidence_gap_target_diagnoses(decision) or list(
                decision.final_diagnoses or []
            )
            corrective_results = await self._maybe_order_critic_exams(
                patient_id=patient_id,
                recommended_exams=recommended_exams,
                exam_results=exam_results,
                collected_info=collected_info,
                candidate_diseases=corrective_targets,
            )
            if corrective_results:
                exam_results.update(corrective_results)
                self.memory_manager.update_exam_results(patient_id, corrective_results)
                self._last_exam_results = dict(exam_results)
                evidence_graph = self.evidence_agent.build_graph(collected_info, exam_results)
                evidence = evidence_graph.bundle
                rag_chunks = self.memory_manager.search_rag(
                    collected_info=collected_info,
                    query=(
                        evidence.to_query()
                        + " 当前鉴别诊断 "
                        + " ".join(decision.final_diagnoses)
                    ),
                    candidate_diseases=decision.final_diagnoses or None,
                )
                decision = self.diagnosis_engine.decide(diagnosis_result, rag_chunks, evidence)
                final_critic = await self.diagnosis_critic.review(
                    decision,
                    evidence,
                    remaining_seconds=self._remaining_case_seconds(),
                    allow_llm=False,
                )
                self._apply_critic_selection(
                    decision,
                    final_critic.selected_diagnoses,
                    final_critic.reason,
                )
                self._restore_legacy_candidate_submission(decision, diagnosis_result, final_critic)
                critic.issues = list(dict.fromkeys(critic.issues + final_critic.issues))

            diagnosis_result = self.diagnosis_engine.apply_to_result(
                diagnosis_result,
                decision,
                evidence,
            )
            diagnosis_result["_critic_review"] = critic.to_dict()
            diagnosis_result["_rag_chunks"] = [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "title": item.get("title"),
                    "score": item.get("score"),
                }
                for item in rag_chunks
            ]
            self._last_diagnosis_audit = {
                "evidence": evidence.to_dict(),
                "evidence_graph": evidence_graph.to_dict(),
                "diagnosis_decision": decision.to_dict(),
                "critic": critic.to_dict(),
                "rag_chunks": diagnosis_result["_rag_chunks"],
                "llm_candidates": llm_candidates,
                "diagnosis_name_resolution": [
                    item.to_dict() for item in llm_resolutions
                ],
            }
        else:
            diagnosis_prompt = self.prompt.build_diagnosis_prompt(
                collected_info=collected_info,
                exam_results=exam_results,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                standard_diseases=self.knowledge.get_disease_catalog_names(),
            )
            messages = [
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content": "请做出诊断并制定治疗方案，以 JSON 格式输出。"},
            ]
            diagnosis_result = await self._llm_generate_diagnosis(messages)
            diagnosis_result = self.evidence_engine.review(
                diagnosis_result,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            if not diagnosis_result.get("_trusted_diagnoses"):
                diagnosis_result = self.structural_agent.review(
                    diagnosis_result,
                    collected_info=collected_info,
                    exam_results=exam_results,
                )

        diagnosis_result = self.quality_agent.review_final_result(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        diagnosis_result = self.treatment_agent.review(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        diagnosis_result = self.treatment_safety.review(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        diagnosis_result = self.quality_agent.review_final_result(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        if (
            decision is not None
            and evidence is not None
            and not self.legacy_candidate_submission
        ):
            diagnosis_result = self._refilter_diagnosis_result(
                diagnosis_result,
                decision,
                evidence,
            )
        filtered_names = self._remove_suppressed_diagnosis_names(
            self._diagnosis_names_from_result(diagnosis_result)
        )
        if filtered_names:
            diagnosis_result["diagnosis"] = filtered_names
        if diagnosis_result.get("_qc_issues"):
            logger.info(f"[质控] 诊疗方案修复/提示: {diagnosis_result['_qc_issues']}")

        logger.info(
            f"[诊断] 诊断结果: {json.dumps(diagnosis_result, ensure_ascii=False, indent=2)}"
        )

        submission_diagnoses = self._remove_suppressed_diagnosis_names(
            self._diagnosis_names_from_result(diagnosis_result)
        )
        if submission_diagnoses:
            diagnosis_result["diagnosis"] = submission_diagnoses

        # 提交诊疗方案
        submit_result = await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=submission_diagnoses or diagnosis_result.get("diagnosis", []),
            treatment_plan=diagnosis_result.get("treatment_plan", ""),
            reasoning=diagnosis_result.get("reasoning", ""),
        )

        final_result = dict(diagnosis_result)
        if isinstance(submit_result, dict):
            for key, value in submit_result.items():
                if value is not None and value != "":
                    final_result[key] = value
        reviewed = self.quality_agent.review_final_result(
            final_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        if submission_diagnoses:
            reviewed["diagnosis"] = submission_diagnoses
        elapsed = round(max(0.0, time.monotonic() - self._case_started_at), 3)
        reviewed["_case_elapsed_seconds"] = elapsed
        reviewed["_case_timed_out"] = False
        if self._last_diagnosis_audit:
            self._last_diagnosis_audit["elapsed_seconds"] = elapsed
            self._last_diagnosis_audit["timed_out"] = False
        return reviewed

    def _planner_candidate_names(self) -> List[str]:
        plan = getattr(self._planner, "current_plan", None) or {}
        names: List[str] = []
        primary = plan.get("primary_hypothesis") or plan.get("primary_diagnosis")
        if primary:
            names.append(str(primary))
        for item in plan.get("differential_diagnoses", []) or []:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = item.get("disease") or item.get("diagnosis") or item.get("name")
            else:
                value = None
            if value:
                names.append(str(value))
        return list(dict.fromkeys(names))[:8]

    @staticmethod
    def _diagnosis_names_from_result(result: Any) -> List[str]:
        if not isinstance(result, dict):
            return []
        values = result.get("diagnosis") or result.get("diagnoses") or []
        if isinstance(values, str):
            values = [values]
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))

    def _refilter_diagnosis_result(self, result: Dict[str, Any], decision, evidence) -> Dict[str, Any]:
        original_names = self._diagnosis_names_from_result(result)
        names = self._remove_suppressed_diagnosis_names(original_names)
        if not names:
            return result
        filtered = self.diagnosis_engine.filter_final_diagnoses(names, decision.candidates)
        if not filtered:
            return result
        filtered_names = [item.diagnosis for item in filtered]
        if filtered_names == original_names:
            return result
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        decision.final_diagnoses = filtered_names
        decision.trusted_diagnoses = [
            name for name in filtered_names
            if score_by_name.get(name)
            and score_by_name[name].score >= self.diagnosis_engine.trusted_threshold
        ]
        if filtered_names and filtered_names[0] in score_by_name:
            decision.confidence = score_by_name[filtered_names[0]].score
        return self.diagnosis_engine.apply_to_result(result, decision, evidence)

    def _remove_suppressed_diagnosis_names(self, names: List[str]) -> List[str]:
        ordered = list(dict.fromkeys(names or []))
        selected = set(ordered)
        suppressed = set()
        for name in ordered:
            entry = self.diagnosis_engine.knowledge.get(name)
            suppressed.update(str(item) for item in entry.get("suppress_diagnoses", []) or [])
            parent = str(entry.get("parent_diagnosis") or "")
            if parent and parent in selected:
                suppressed.add(parent)
        return [name for name in ordered if name not in suppressed]

    def _apply_critic_selection(
        self,
        decision,
        selected_diagnoses: List[str],
        reason: str,
    ) -> None:
        if not selected_diagnoses:
            return
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        validated = [
            name for name in selected_diagnoses
            if name in score_by_name
            and not score_by_name[name].hard_contradiction
            and score_by_name[name].trusted
        ]
        if not validated:
            return
        strong_evidence = [
            item.diagnosis for item in decision.candidates
            if item.diagnosis in decision.trusted_diagnoses
            and item.support_score >= 0.8
            and not item.hard_contradiction
        ]
        selected_best = max(
            (score_by_name[name].score for name in validated if name in score_by_name),
            default=0.0,
        )
        protected_causal = [
            item.diagnosis for item in decision.candidates[:5]
            if item.trusted
            and self._is_etiology_priority_candidate(item)
            and item.score >= self.diagnosis_engine.trusted_threshold
            and item.score >= selected_best - self.diagnosis_engine.margin_threshold
        ]
        # LLM Critic may reorder ambiguous candidates, but cannot discard a
        # diagnosis already established by strong deterministic evidence.
        filtered = self.diagnosis_engine.filter_final_diagnoses(
            list(dict.fromkeys(protected_causal + strong_evidence + validated)),
            decision.candidates,
        )
        if not filtered:
            return
        decision.final_diagnoses = [item.diagnosis for item in filtered]
        decision.trusted_diagnoses = [
            name for name in decision.final_diagnoses
            if score_by_name[name].score >= self.diagnosis_engine.trusted_threshold
        ]
        decision.confidence = score_by_name[decision.final_diagnoses[0]].score
        decision.differential_only_diagnoses = self.diagnosis_engine.differential_only_details(
            decision.candidates
        )
        if reason:
            extra = self.diagnosis_engine.differential_only_reasoning(decision.candidates)
            suffix = "。提交前审查：" + str(reason).rstrip("。") + "。"
            if extra:
                suffix += extra
            decision.evidence_reasoning = decision.evidence_reasoning.rstrip("。") + suffix

    def _restore_legacy_candidate_submission(
        self,
        decision,
        llm_result: Dict[str, Any],
        critic,
    ) -> None:
        """Restore the earlier broad-submission behavior for evidence-backed candidates.

        This keeps the Evidence-first ranking, but lets an LLM primary diagnosis or
        Critic-supported candidate re-enter the final submission when it is already
        represented in the scored candidate table and has no hard contradiction.
        """
        if not self.legacy_candidate_submission or not decision:
            return
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        primary_names = self._resolved_llm_primary_names(llm_result)
        supporting_names = self._resolved_llm_candidate_names(llm_result)
        supporting_names.extend(self._resolved_critic_names(critic))

        primary_eligible = [
            name for name in primary_names
            if self._legacy_submission_eligible(score_by_name.get(name))
        ]
        support_eligible = [
            name for name in supporting_names
            if self._legacy_submission_eligible(score_by_name.get(name))
        ]

        if primary_eligible and self._should_prefer_legacy_primary(primary_eligible, score_by_name):
            names = list(primary_eligible)
        else:
            names = list(decision.final_diagnoses or [])

        for name in support_eligible:
            if name not in names:
                names.append(name)
            if len(names) >= self.diagnosis_engine.max_final_diagnoses:
                break

        if not names:
            return
        names = self._remove_suppressed_diagnosis_names(names)
        filtered = [
            score_by_name[name] for name in names
            if name in score_by_name
            and self._legacy_submission_eligible(score_by_name[name])
        ][: self.diagnosis_engine.max_final_diagnoses]
        if not filtered:
            return
        filtered_names = [item.diagnosis for item in filtered]
        if filtered_names == list(decision.final_diagnoses or []):
            return
        decision.final_diagnoses = filtered_names
        decision.trusted_diagnoses = [
            item.diagnosis for item in filtered
            if item.score >= self.diagnosis_engine.trusted_threshold
        ]
        decision.confidence = filtered[0].score
        suffix = (
            "。提交前审查：已恢复旧版宽松候选提交逻辑；"
            "LLM/Critic 已提出且证据候选中无硬反证的标准诊断被保留。"
        )
        if suffix not in decision.evidence_reasoning:
            decision.evidence_reasoning = decision.evidence_reasoning.rstrip("。") + suffix

    def _resolved_llm_primary_names(self, result: Dict[str, Any]) -> List[str]:
        if not isinstance(result, dict):
            return []
        values = result.get("diagnosis") or result.get("diagnoses") or []
        if isinstance(values, str):
            values = [values]
        return self._resolve_diagnosis_values(values)

    def _resolved_llm_candidate_names(self, result: Dict[str, Any]) -> List[str]:
        if not isinstance(result, dict):
            return []
        values: List[Any] = []
        for key in (
            "diagnosis_candidates",
            "open_diagnosis_candidates",
            "candidate_diagnoses",
            "differential_diagnoses",
        ):
            current = result.get(key)
            if current:
                values.extend(current if isinstance(current, list) else [current])
        return self._resolve_diagnosis_values(values)

    def _resolved_critic_names(self, critic) -> List[str]:
        values: List[Any] = []
        values.extend(list(getattr(critic, "selected_diagnoses", []) or []))
        values.extend(self._extract_allowed_diagnoses_from_text(getattr(critic, "reason", "") or ""))
        return self._resolve_diagnosis_values(values)

    def _resolve_diagnosis_values(self, values: List[Any]) -> List[str]:
        names: List[str] = []
        for value in values or []:
            confidence = 1.0
            if isinstance(value, dict):
                raw = value.get("name") or value.get("diagnosis") or value.get("disease")
                try:
                    confidence = float(value.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    confidence = 1.0
            else:
                raw = value
            if confidence < 0.45:
                continue
            resolved = self.diagnosis_engine.resolver.resolve(raw, model_confidence=confidence)
            if resolved.canonical_name and resolved.canonical_name not in names:
                names.append(resolved.canonical_name)
        return names

    def _extract_allowed_diagnoses_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        names: List[str] = []
        for name in sorted(self.diagnosis_engine.knowledge.allowed_names, key=len, reverse=True):
            if name not in text:
                continue
            if self._diagnosis_mention_is_negated(text, name):
                continue
            if name not in names:
                names.append(name)
        return names[: self.diagnosis_engine.max_final_diagnoses * 2]

    @staticmethod
    def _diagnosis_mention_is_negated(text: str, name: str) -> bool:
        index = text.find(name)
        if index < 0:
            return False
        window = text[max(0, index - 18): index + len(name) + 28]
        negators = (
            "不支持", "排除", "不选", "未选", "缺乏", "无", "不能解释",
            "证据极弱", "可能性低", "不符合", "否定",
        )
        return any(token in window for token in negators)

    def _legacy_submission_eligible(self, candidate) -> bool:
        if candidate is None:
            return False
        return (
            candidate.trusted
            and bool(candidate.matched_evidence)
            and not candidate.hard_contradiction
            and candidate.score >= self.diagnosis_engine.differential_threshold
        )

    def _should_prefer_legacy_primary(
        self,
        primary_names: List[str],
        score_by_name: Dict[str, Any],
    ) -> bool:
        for name in primary_names:
            candidate = score_by_name.get(name)
            if not candidate:
                continue
            dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
            if dtype in {"etiology", "metabolic", "structural"}:
                return True
            if float(getattr(candidate, "specificity", 0.0) or 0.0) >= 0.85:
                return True
        return False

    def _recommend_evidence_gap_exams(
        self,
        decision,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> List[str]:
        if not decision:
            return []
        targets = self._evidence_gap_target_diagnoses(decision)
        if not targets:
            return []

        proposed: List[str] = []
        for name in targets:
            entry = self.diagnosis_engine.knowledge.get(name)
            for exam in entry.get("discriminating_exams", []) or []:
                text = str(exam).strip()
                if text and text not in proposed:
                    proposed.append(text)

        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=targets,
            proposed_items=proposed,
            existing_results=exam_results,
        )
        ordered_items = list(dict.fromkeys(proposed + list(strategy.get("items", []) or [])))
        return self.exam_agent.prepare_order_items(
            ordered_items,
            collected_info=collected_info,
            candidate_diseases=targets,
            existing_results=exam_results,
            max_items=self.diagnosis_critic.max_corrective_exam_items,
            add_strong_verification=False,
        )

    def _evidence_gap_target_diagnoses(self, decision) -> List[str]:
        by_name = {item.diagnosis: item for item in decision.candidates}
        targets: List[str] = []

        close_margin = getattr(self.diagnosis_engine, "etiology_close_margin", 0.12)
        coverage_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_coverage_threshold",
            0.32,
        )
        residual_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_residual_threshold",
            0.72,
        )
        selected = by_name.get((decision.final_diagnoses or [""])[0])
        gap_candidates = []
        for item in decision.candidates:
            if (
                item.required_gaps
                and item.matched_evidence
                and not item.hard_contradiction
                and self._is_etiology_priority_candidate(item)
                and (
                    getattr(item, "coverage_score", 0.0) >= coverage_threshold
                    or getattr(item, "residual_score", 1.0) <= residual_threshold
                    or item.source_prior >= 0.45
                    or (
                        selected is not None
                        and item.score >= max(0.0, selected.score - close_margin)
                    )
                )
            ):
                gap_candidates.append(item)
        gap_candidates.sort(
            key=lambda item: (
                getattr(item, "coverage_score", 0.0),
                1.0 - getattr(item, "residual_score", 1.0),
                item.score,
                item.specificity,
            ),
            reverse=True,
        )

        has_gap = bool(
            decision.low_confidence
            or decision.unexplained_evidence
            or gap_candidates
            or self._has_close_etiology_candidate(decision)
        )
        if not has_gap:
            return []

        for item in gap_candidates:
            if item.diagnosis not in targets:
                targets.append(item.diagnosis)

        for name in (decision.final_diagnoses or [])[:1]:
            if name in by_name and name not in targets:
                targets.append(name)

        priority_candidates = [
            item
            for item in decision.candidates
            if item.matched_evidence
            and not item.hard_contradiction
            and self._is_etiology_priority_candidate(item)
        ]
        priority_candidates.sort(
            key=lambda item: (
                getattr(item, "coverage_score", 0.0),
                1.0 - getattr(item, "residual_score", 1.0),
                item.score,
            ),
            reverse=True,
        )
        if priority_candidates and priority_candidates[0].diagnosis not in targets:
            targets.append(priority_candidates[0].diagnosis)

        unexplained = set(decision.unexplained_evidence or [])
        if unexplained:
            for item in decision.candidates:
                if item.hard_contradiction:
                    continue
                if unexplained & set(item.matched_evidence or []):
                    if item.diagnosis not in targets:
                        targets.append(item.diagnosis)
                    break

        limit = getattr(self.diagnosis_engine, "max_evidence_gap_targets", 2)
        return targets[: max(1, int(limit or 2))]

    def _has_close_etiology_candidate(self, decision) -> bool:
        if not decision.candidates or not decision.final_diagnoses:
            return False
        by_name = {item.diagnosis: item for item in decision.candidates}
        selected = by_name.get(decision.final_diagnoses[0])
        if not selected:
            return False
        close_margin = getattr(self.diagnosis_engine, "etiology_close_margin", 0.12)
        coverage_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_coverage_threshold",
            0.32,
        )
        residual_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_residual_threshold",
            0.72,
        )
        for item in decision.candidates:
            if item.diagnosis == selected.diagnosis:
                continue
            if (
                item.matched_evidence
                and not item.hard_contradiction
                and self._is_etiology_priority_candidate(item)
                and (
                    item.required_gaps
                    or item.source_prior >= 0.45
                    or getattr(item, "coverage_score", 0.0) >= coverage_threshold
                    or getattr(item, "residual_score", 1.0) <= residual_threshold
                    or item.score >= max(0.0, selected.score - close_margin)
                )
            ):
                return True
        return False

    @staticmethod
    def _is_etiology_priority_candidate(candidate) -> bool:
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        specificity = float(getattr(candidate, "specificity", 0.0) or 0.0)
        return dtype in {"etiology", "metabolic", "structural"} or specificity >= 0.85

    async def _maybe_order_critic_exams(
        self,
        patient_id: str,
        recommended_exams: List[str],
        exam_results: Dict[str, Any],
        collected_info: Optional[Dict[str, Any]] = None,
        candidate_diseases: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        if (
            not recommended_exams
            or self._remaining_case_seconds() < self.diagnosis_critic.corrective_exam_min_seconds
        ):
            return {}
        planner = self._get_planner()
        if planner.exam_rounds >= self.max_exam_rounds:
            return {}
        items = self.exam_agent.prepare_order_items(
            recommended_exams,
            collected_info=collected_info or {},
            candidate_diseases=candidate_diseases,
            existing_results=exam_results,
            max_items=self.diagnosis_critic.max_corrective_exam_items,
        )
        if not items:
            return {}
        try:
            response = await self.actions.order_examination(
                patient_id=patient_id,
                items=items,
                reason="提交前诊断审查发现低置信或未解释证据，补充最具鉴别价值的检查。",
            )
        except Exception as exc:
            logger.warning("[DiagnosisCritic] corrective examination failed: %s", exc)
            return {}
        new_results: Dict[str, Any] = {}
        for exam_name, exam_data in (response or {}).get("results", {}).items():
            if isinstance(exam_data, dict) and exam_data.get("status") != "invalid":
                new_results[exam_name] = exam_data
        if new_results:
            planner.exam_rounds += 1
            planner._record_action(
                "order_examination",
                ",".join(new_results.keys()),
                "diagnosis_critic_corrective_exam",
            )
        return new_results

    # ============ 反思并保存经验 ============

    def _save_fast_reflection(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Save a lightweight training note without an extra reflection LLM call."""
        diagnosis_accuracy = report.get("diagnosisAccuracy", report.get("diagnosis_accuracy", 0))
        exam_precision = report.get("examinationPrecision", report.get("examination_precision", 0))
        treatment_score = report.get("treatmentOverallScore", report.get("treatment_overall_score", 0))
        reflection = (
            "快速训练反思："
            f"诊断准确率={diagnosis_accuracy}，"
            f"检查精确率={exam_precision}，"
            f"治疗评分={treatment_score}。"
            "本轮为快速路径，已保留问诊、检查和评估结果供后续检索。"
        )
        error_types = self._classify_diagnosis_errors(report)
        audit = self._last_diagnosis_audit or {}
        self.memory.save_case_experience(
            patient_id=patient_id,
            report=report,
            reflection=reflection,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence"),
            diagnosis_decision=audit.get("diagnosis_decision"),
            error_types=error_types,
        )
        self.memory.save_diagnostic_replay(
            patient_id=patient_id,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence") or {},
            diagnosis_decision=audit.get("diagnosis_decision") or {},
            report=report,
            error_types=error_types,
            llm_candidates=audit.get("llm_candidates") or [],
            rag_chunks=audit.get("rag_chunks") or [],
            case_audit=audit,
        )
        self._record_exam_alias_feedback(patient_id, report, exam_results)
        self._record_diagnostic_rule_feedback(patient_id, report, collected_info, exam_results)
        logger.info("[Reflection] fast reflection saved for %s", patient_id)

    def _classify_diagnosis_errors(self, report: Dict[str, Any]) -> List[str]:
        """Classify evaluation failures into actionable diagnosis subsystems."""
        errors: List[str] = []
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        expected = detail.get("expected") or report.get("finalDiagnosis") or []
        submitted = detail.get("submitted") or report.get("diagnosis") or []
        if isinstance(expected, str):
            expected = [expected]
        if isinstance(submitted, str):
            submitted = [submitted]
        audit = self._last_diagnosis_audit or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") or []
        candidate_names = [str(item.get("diagnosis")) for item in candidates if isinstance(item, dict)]
        final_names = [str(item) for item in decision.get("final_diagnoses", []) or []]
        for name in expected:
            name = str(name)
            if not self.diagnosis_engine.knowledge.is_allowed(name):
                errors.append("namespace_error")
            elif name not in candidate_names[:5]:
                errors.append("candidate_recall_error")
            elif name not in final_names:
                errors.append("candidate_ranking_error")
            matched = next(
                (
                    item.get("matched_evidence") or []
                    for item in candidates
                    if isinstance(item, dict) and item.get("diagnosis") == name
                ),
                [],
            )
            if not matched:
                errors.append("evidence_extraction_failure")
        if any(not self.diagnosis_engine.knowledge.is_allowed(item) for item in submitted):
            errors.append("namespace_error")
        examination_detail = report.get("examinationDetail") or report.get("examination_detail") or {}
        try:
            if float(examination_detail.get("coverage", 1.0)) < 0.5:
                errors.append("insufficient_examination")
        except (TypeError, ValueError):
            pass
        treatment_detail = report.get("treatmentDetail") or report.get("treatment_detail") or {}
        try:
            if float(treatment_detail.get("safety", 1.0)) < 0.8:
                errors.append("treatment_safety_failure")
        except (TypeError, ValueError):
            pass
        return list(dict.fromkeys(errors))

    def _record_exam_alias_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Collect exam alias evidence from training feedback without breaking training."""
        try:
            if not hasattr(self.knowledge, "record_exam_alias_feedback"):
                return
            submitted_items = list((exam_results or {}).keys())
            stats = self.knowledge.record_exam_alias_feedback(
                patient_id=patient_id,
                report=report,
                submitted_items=submitted_items,
            )
            if stats.get("pending") or stats.get("promoted"):
                logger.info(
                    "[ExamAlias] feedback recorded: pending=%s, promoted=%s",
                    stats.get("pending", 0),
                    stats.get("promoted", 0),
                )
        except Exception as exc:
            logger.warning("[ExamAlias] feedback collection failed: %s", exc)

    def _record_diagnostic_rule_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Collect evaluation feedback as shadow evidence for replay validation."""
        try:
            audit = self._last_diagnosis_audit or {}
            stats = self.diagnostic_learning.record_feedback(
                patient_id=patient_id,
                report=report,
                evidence=audit.get("evidence") or {},
                diagnosis_decision=audit.get("diagnosis_decision") or {},
                error_types=self._classify_diagnosis_errors(report),
            )
            if stats.get("pending") or stats.get("updated"):
                logger.info(
                    "[EvidenceRules] shadow candidates added=%s updated=%s",
                    stats.get("pending", 0),
                    stats.get("updated", 0),
                )
        except Exception as exc:
            logger.warning("[EvidenceRules] feedback collection failed: %s", exc)

    async def _reflect_and_save(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """反思评估结果并保存经验。

        Args:
            patient_id: 患者 ID
            report: 评估报告
            collected_info: 收集到的患者信息
            exam_results: 检查结果
        """
        # 三源 RAG：knowledge + memory + 当前病例
        _sym_r = (collected_info or {}).get("symptoms") or []
        _final_dx = None
        try:
            _final_dx = (report or {}).get("finalDiagnosis") or (report or {}).get("diagnosis")
        except Exception:
            _final_dx = None
        _cands_r = [_final_dx] if _final_dx else None
        try:
            knowledge_context_r = self.knowledge.build_rag_context(_sym_r, _cands_r)
        except Exception:
            knowledge_context_r = ""
        try:
            if getattr(self, "memory_manager", None):
                memory_context_r = self.memory_manager.build_semantic_context(
                    collected_info, _cands_r
                )
                if memory_context_r:
                    knowledge_context_r = memory_context_r
        except Exception:
            pass

        # 多维召回历史经验
        try:
            relevant_exp_r = self._get_cached_experience(collected_info) or ""
        except Exception:
            relevant_exp_r = ""

        # 构建反思 prompt
        reflect_prompt = self.prompt.build_reflection_prompt(
            report=report,
            collected_info=collected_info,
            exam_results=exam_results,
            knowledge_context=knowledge_context_r,
            relevant_experience=relevant_exp_r,
        )
        messages = [
            {"role": "system", "content": reflect_prompt},
            {"role": "user", "content": "请进行结构化反思，总结经验教训。"},
        ]

        # 使用 LLM 生成反思总结
        reflection = await self._llm_chat(messages, temperature=0.5)

        if not reflection:
            # 回退到简单反思
            diagnosis_accuracy = report.get("diagnosisAccuracy", 0)
            exam_precision = report.get("examinationPrecision", 0)
            treatment_score = report.get("treatmentOverallScore", 0)
            reflection = (
                f"诊断准确率: {diagnosis_accuracy}, "
                f"检查精确率: {exam_precision}, "
                f"治疗评分: {treatment_score}。"
            )
            if diagnosis_accuracy < 0.8:
                reflection += " 需要改进问诊策略，收集更多鉴别诊断信息。"
            if exam_precision < 0.8:
                reflection += " 需要优化检查选择，减少不必要的检查。"
            if treatment_score < 0.8:
                reflection += " 需要改进治疗方案，提高个性化和有效性。"

        # 保存到记忆。失败病例仅作为纠错教训渲染，并单独保留完整诊断回放。
        error_types = self._classify_diagnosis_errors(report)
        audit = self._last_diagnosis_audit or {}
        self.memory.save_case_experience(
            patient_id=patient_id,
            report=report,
            reflection=reflection,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence"),
            diagnosis_decision=audit.get("diagnosis_decision"),
            error_types=error_types,
        )
        self.memory.save_diagnostic_replay(
            patient_id=patient_id,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence") or {},
            diagnosis_decision=audit.get("diagnosis_decision") or {},
            report=report,
            error_types=error_types,
            llm_candidates=audit.get("llm_candidates") or [],
            rag_chunks=audit.get("rag_chunks") or [],
            case_audit=audit,
        )
        self._record_exam_alias_feedback(patient_id, report, exam_results)
        self._record_diagnostic_rule_feedback(patient_id, report, collected_info, exam_results)

        # ============ 自迭代闭环 ============
        # 1) 缺陷检测 → 2) 编译为策略补丁 → 3) 反馈本例 ΔScore
        try:
            if self.detector is not None and self.policy_store is not None:
                # 步骤1：缺陷检测（规则通道 + LLM 归因通道合并；后者受 config 开关与 llm_chat 注入双重控制）
                _defects = await self.detector.detect_all(
                    report=report,
                    collected_info=collected_info,
                    exam_results=exam_results,
                    action_history=(self._planner.action_history
                                    if self._planner else None),
                    use_llm=self.self_improve_use_llm_attribute,
                )
                # 步骤2：编译补丁（默认 shadow，收敛后由 audit 提权到 active）
                _emitted = self.policy_store.emit_from_defects(
                    _defects, default_status="shadow"
                )
                # 步骤3：本例用到的补丁 → 结合 ΔScore 反馈
                _used_ids = []
                if self._planner is not None:
                    _used_ids = list(self._planner._last_used_patch_ids or [])
                if _used_ids:
                    _overall = _overall_score(report)
                    _baseline = self.memory.get_score_baseline(
                        final_dx=(report or {}).get("finalDiagnosis")
                                 or (report or {}).get("diagnosis")
                    ) if hasattr(self.memory, "get_score_baseline") else 0.7
                    _delta = None
                    if _overall is not None and _baseline is not None:
                        _delta = _overall - _baseline
                    self.policy_store.record_outcome(_used_ids, _delta or 0.0)
                    logger.info(
                        f"[自迭代] 本例补丁反馈: used={len(_used_ids)}, "
                        f"score={_overall}, baseline={_baseline}, delta={_delta}"
                    )
                # 步骤4：每 5 例做一次 audit（元迭代）
                _n_cases = len(getattr(self.memory, "notes", []) or [])
                if (
                    not self.freeze_active_learning
                    and _n_cases > 0
                    and _n_cases % 5 == 0
                ):
                    self.policy_store.audit()

                if _defects:
                    logger.info(
                        f"[自迭代] 反思阶段识别缺陷 {len(_defects)} 项，"
                        f"emit 补丁 {len(_emitted)} 项 "
                        f"(库大小={len(self.policy_store.patches)})"
                    )
        except Exception as _e:
            logger.warning(f"[自迭代] 反思阶段闭环失败(不影响主流程): {_e}")

        logger.info(f"[反思] 患者 {patient_id} 反思已保存")

    # ============ 思考链 ============

    async def _think(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        phase: str,
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """思考当前诊疗状态，生成鉴别诊断和下一步行动指引。

        合并了充分性判断功能：返回结果中包含 is_sufficient 字段。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            phase: 当前阶段（"inquiry" 或 "examination"）
            relevant_experience: 相关历史经验

        Returns:
            思考结果，包含：
            - differential_diagnosis: 鉴别诊断列表
            - key_unknowns: 关键未知项
            - is_sufficient: 信息是否足够
            - next_action: 下一步行动建议
            - action_reasoning: 行动理由
        """
        # 前置知识库召回：基于症状检索候选疾病，缩小 LLM 搜索空间
        knowledge_context = ""
        try:
            symptoms = collected_info.get("symptoms", []) or []
            if symptoms and getattr(self, "memory_manager", None):
                memory_context = self.memory_manager.build_semantic_context(
                    collected_info
                )
                if memory_context:
                    knowledge_context = memory_context
                    logger.info("[思考] 已注入结构化语义记忆上下文")
            elif symptoms and getattr(self, "knowledge", None):
                knowledge_context = self.knowledge.build_rag_context(symptoms=symptoms)
                if knowledge_context:
                    logger.info("[思考] 已注入知识库 RAG 上下文")
        except Exception as e:
            logger.warning(f"[思考] 知识库召回失败: {e}")

        thinking_prompt = self.prompt.build_thinking_prompt(
            collected_info=collected_info,
            exam_results=exam_results or {},
            chat_history=chat_history,
            phase=phase,
            relevant_experience=relevant_experience,
            knowledge_context=knowledge_context,
        )
        messages = [
            {"role": "system", "content": thinking_prompt},
            {"role": "user", "content": "请进行临床推理分析。"},
        ]

        result = await self._llm_chat_json(messages, temperature=0.3)

        if result and "differential_diagnosis" in result:
            dd_names = [d.get("diagnosis", "?") for d in result.get("differential_diagnosis", [])]
            logger.info(f"[思考] 阶段={phase}, 鉴别诊断: {dd_names}")
            logger.info(f"[思考] 关键未知项: {result.get('key_unknowns', [])}")
            logger.info(f"[思考] 信息充分: {result.get('is_sufficient', False)}")
            logger.info(f"[思考] 建议下一步: {result.get('next_action', '')}")
            return result

        logger.warning("[思考] LLM 思考失败，返回空思考结果")
        return {}

    # ============ LLM 辅助方法 ============

    async def _llm_chat(
        self, messages: List[Dict[str, str]], temperature: float = None
    ) -> str:
        """调用 LLM 进行对话。

        Args:
            messages: 消息列表
            temperature: 生成温度

        Returns:
            LLM 响应文本
        """
        if not self._can_call_llm("chat"):
            return ""
        try:
            if self.log_llm_prompts:
                logger.debug(f"[LLM] Prompt: {json.dumps(messages, ensure_ascii=False)[:500]}...")

            response = await self.llm.chat(messages, temperature=temperature)
            self._bump_llm_counter("chat")

            if self.log_llm_prompts:
                logger.debug(f"[LLM] Response: {response[:500]}...")

            return response.strip()

        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            return ""

    async def _llm_chat_json(
        self, messages: List[Dict[str, str]], temperature: float = None
    ) -> Dict[str, Any]:
        """调用 LLM 并解析 JSON 响应。

        Args:
            messages: 消息列表
            temperature: 生成温度

        Returns:
            解析后的 JSON 字典
        """
        if not self._can_call_llm("json"):
            return {}
        try:
            result = await self.llm.chat_json(messages, temperature=temperature)
            self._bump_llm_counter("json")
            return result
        except Exception as e:
            logger.error(f"[LLM] JSON 调用失败: {e}")
            return {}

    async def _extract_patient_info(
        self, patient_response: str, existing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 LLM 从患者回复中提取结构化信息。

        Args:
            patient_response: 患者回复文本
            existing_info: 已有的患者信息

        Returns:
            更新后的患者信息
        """
        # 构建 LLM 信息提取 prompt
        extraction_prompt = self.prompt.build_info_extraction_prompt(
            patient_response=patient_response,
            existing_info=existing_info,
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": "请从患者回复中提取结构化信息。"},
        ]

        # 调用 LLM 提取信息
        extracted = await self._llm_chat_json(messages, temperature=0.3)

        if not extracted or "raw_response" in extracted:
            # LLM 提取失败，回退到关键词提取
            logger.warning("[信息提取] LLM 提取失败，回退到关键词提取")
            return self._fallback_parse_patient_response(patient_response, existing_info)

        # 合并提取结果到已有信息
        info = existing_info.copy()

        # 合并各字段（保留已有非空值）
        for key in [
            "chief_complaint", "present_illness", "past_history",
            "medication_history", "allergy_history", "family_history",
            "personal_history",
        ]:
            value = extracted.get(key, "")
            if value and (not info.get(key) or info.get(key) == ""):
                info[key] = value

        # 合并症状列表
        new_symptoms = extracted.get("symptoms", [])
        existing_symptoms = info.get("symptoms", [])
        info["symptoms"] = list(set(existing_symptoms + new_symptoms))

        # 合并症状详情
        new_details = extracted.get("symptom_details", {})
        existing_details = info.get("symptom_details", {})
        existing_details.update(new_details)
        info["symptom_details"] = existing_details

        # 保留原始回复
        if not info.get("raw_responses"):
            info["raw_responses"] = []
        info["raw_responses"].append(patient_response)

        return info

    def _fallback_parse_patient_response(
        self, response: str, existing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """回退的患者回复解析（关键词提取）。

        Args:
            response: 患者回复文本
            existing_info: 已有的患者信息

        Returns:
            更新后的患者信息
        """
        info = existing_info.copy()

        if not info.get("raw_responses"):
            info["raw_responses"] = []
        info["raw_responses"].append(response)

        # 关键词提取
        symptoms = info.get("symptoms", [])
        symptom_keywords = [
            "发热", "咳嗽", "咳痰", "胸痛", "腹痛", "腹泻",
            "头痛", "头晕", "心悸", "胸闷", "气短", "恶心",
            "呕吐", "乏力", "食欲不振", "失眠", "水肿",
        ]
        symptom_keywords.extend([
            "呼吸困难", "呼吸急促", "气促", "喘息", "喘不上气", "发绀",
            "胸口闷", "出汗", "畏寒", "寒战", "便秘", "停经", "月经异常",
            "多饮", "多尿", "尿频", "尿急", "尿痛", "关节痛", "皮疹",
        ])
        for keyword in symptom_keywords:
            if keyword in response and keyword not in symptoms:
                symptoms.append(keyword)
        info["symptoms"] = symptoms

        return info

    async def _check_info_sufficient(
        self, collected_info: Dict[str, Any], ask_rounds: int, max_ask_rounds: int
    ) -> bool:
        """使用 LLM 判断信息是否已足够。

        Args:
            collected_info: 已收集的信息
            ask_rounds: 当前问诊轮次
            max_ask_rounds: 最大问诊轮次

        Returns:
            是否信息足够
        """
        # 快速判断：如果已达最大轮次，直接返回 True
        if ask_rounds >= max_ask_rounds:
            return True

        # 快速判断：如果关键信息缺失，直接返回 False
        if not collected_info.get("chief_complaint") and not collected_info.get("symptoms"):
            return False

        # 使用 LLM 判断
        prompt = self.prompt.build_info_sufficiency_prompt(
            collected_info=collected_info,
            ask_rounds=ask_rounds,
            max_ask_rounds=max_ask_rounds,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请判断信息是否足够。"},
        ]

        result = await self._llm_chat_json(messages, temperature=0.3)

        if result and "is_sufficient" in result:
            is_sufficient = result["is_sufficient"]
            missing = result.get("missing_aspects", [])
            if missing:
                logger.info(f"[问诊] 信息缺口: {missing}")
            return bool(is_sufficient)

        # 回退：简单判断
        return len(collected_info.get("symptoms", [])) >= 3

    async def _check_exam_sufficient(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        exam_rounds: int,
        max_exam_rounds: int,
    ) -> bool:
        """使用 LLM 判断检查是否已足够。

        Args:
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            exam_rounds: 当前检查轮次
            max_exam_rounds: 最大检查轮次

        Returns:
            是否检查足够
        """
        # 快速判断：如果已达最大轮次，直接返回 True
        if exam_rounds >= max_exam_rounds:
            return True

        # 使用 LLM 判断
        prompt = self.prompt.build_exam_sufficiency_prompt(
            collected_info=collected_info,
            exam_results=exam_results,
            exam_rounds=exam_rounds,
            max_exam_rounds=max_exam_rounds,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请判断检查是否足够。"},
        ]

        result = await self._llm_chat_json(messages, temperature=0.3)

        if result and "is_sufficient" in result:
            is_sufficient = result["is_sufficient"]
            additional = result.get("additional_exams_needed", [])
            if additional:
                logger.info(f"[检查] 建议补充检查: {additional}")
            return bool(is_sufficient)

        # 回退：至少有一项检查结果
        return len(exam_results) >= 1

    async def _llm_generate_examination_items(
        self, messages: List[Dict[str, str]], collected_info: Dict[str, Any]
    ) -> List[str]:
        """使用 LLM 生成检查项目列表。

        Args:
            messages: LLM 消息列表
            collected_info: 已收集的患者信息

        Returns:
            检查项目列表
        """
        result = await self._llm_chat_json(messages, temperature=0.3)

        if isinstance(result, list):
            return [str(item) for item in result if item]

        if isinstance(result, dict):
            # 尝试从字典中提取列表
            for key in ["items", "examinations", "exams", "checks"]:
                if key in result and isinstance(result[key], list):
                    return [str(item) for item in result[key] if item]

        # 回退到规则推荐
        logger.warning("[检查] LLM 生成检查项目失败，回退到规则推荐")
        return self._fallback_generate_examination_items(collected_info)

    def _fallback_generate_examination_items(
        self, collected_info: Dict[str, Any]
    ) -> List[str]:
        """回退的检查项目推荐（基于规则）。

        Args:
            collected_info: 已收集的患者信息

        Returns:
            检查项目列表
        """
        items = []
        symptoms = collected_info.get("symptoms", [])

        if any(s in str(symptoms) for s in ["发热", "咳嗽", "咳痰", "胸痛"]):
            items.extend(["血常规", "胸部CT"])
        if any(s in str(symptoms) for s in ["腹痛", "腹泻", "恶心"]):
            items.extend(["血常规", "腹部B超"])
        if any(s in str(symptoms) for s in ["头痛", "头晕", "意识障碍"]):
            items.extend(["血常规", "头颅CT"])
        if any(s in str(symptoms) for s in ["心悸", "胸闷", "气短"]):
            items.extend(["心电图", "血常规"])

        if not items:
            items = ["血常规", "尿常规"]

        return items

    async def _llm_generate_diagnosis(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """使用 LLM 生成诊断和治疗方案。

        Args:
            messages: LLM 消息列表

        Returns:
            包含 diagnosis, treatment_plan, reasoning 的字典
        """
        result = await self._llm_chat_json(messages, temperature=0.5)

        if result and "diagnosis" in result:
            return result

        # 回退
        logger.warning("[诊断] LLM 生成诊断失败，返回标准目录内兜底诊断")
        return {
            "diagnosis": ["上呼吸道感染"],
            "treatment_plan": "建议进一步完善检查，明确诊断后制定治疗方案。",
            "reasoning": "基于当前问诊和检查信息，暂未明确诊断，需进一步检查。",
        }


if __name__ == "__main__":
    # 当通过 python -m agent 运行时，启动 HTTP 服务
    # 注意：直接运行 python agent/agent.py 不支持（相对导入限制）
    # 请使用 python -m agent 代替
    from agent.server import run_server

    port = int(os.environ.get("PORT", "7860"))
    run_server(host="0.0.0.0", port=port)
