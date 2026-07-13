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

import json
import logging
import os
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

        # 静态医学知识库（症状倒排 + 检查规范化 + RAG）
        ref_dir = config.get("ref_data_dir", "data/ref_data")
        self.knowledge = KnowledgeBase(ref_dir=ref_dir)
        self.exam_agent = ExamStrategyAgent(self.knowledge)
        self.inquiry_agent = InquiryStrategyAgent(self.knowledge)
        self.quality_agent = QualityAgent(self.knowledge)
        self.treatment_agent = TreatmentStrategyAgent(self.knowledge)

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
                logger.info(
                    f"[自迭代] 已启用 detector + policy_store "
                    f"(现有补丁 {len(self.policy_store.patches)} 项)"
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
            # 注入策略补丁库（若 agent 已启用）
            if getattr(self, "policy_store", None) is not None:
                self._planner.policy_store = self.policy_store
        return self._planner

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
        if strategy.get("added_required"):
            logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
        if strategy.get("invalid_items"):
            logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
        exam_items = strategy.get("items", [])
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

    async def train(self, patient_id: str) -> None:
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
        final_result = await self._execute_with_planner(patient_id)

        # 训练阶段：获取评估报告并反思
        try:
            report = await self.actions.evaluation(
                patient_id=patient_id, final_result=final_result
            )
            logger.info(
                f"[Train] 患者 {patient_id} 评估报告: "
                f"{json.dumps(report, ensure_ascii=False, indent=2)}"
            )

            # 反思并保存经验
            await self._reflect_and_save(
                patient_id, report,
                self._last_collected_info,
                self._last_exam_results,
            )
        except Exception as e:
            logger.warning(f"[Train] 获取评估报告失败: {e}")

        # 输出记忆统计
        stats = self.memory.get_statistics()
        logger.info(f"[Train] 记忆统计: {json.dumps(stats, ensure_ascii=False)}")
        logger.info(
            f"[Train] LLM 调用统计: 总{self._llm_call_count}次, "
            f"明细={json.dumps(self._llm_call_by_kind, ensure_ascii=False)}"
        )
        logger.info(f"[Train] 完成训练患者: {patient_id}")

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
        final_result = await self._execute_with_planner(patient_id)

        # 保存测试结果供 run_test 收集
        planner = self._get_planner()
        _rounds = len([a for a in planner.action_history if a["type"] == "ask_patient"])
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
            if strategy.get("added_required"):
                logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
            if strategy.get("invalid_items"):
                logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
            exam_items = strategy.get("items", [])

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
        # 基于症状检索相关经验
        symptoms = collected_info.get("symptoms", [])
        if symptoms:
            relevant_experience = self.memory.search_relevant_experience(symptoms, top_k=3)

        # 构建诊断 prompt
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

        # 使用 LLM 生成诊断和治疗方案
        diagnosis_result = await self._llm_generate_diagnosis(messages)
        conversation_rounds = len([m for m in chat_history if m.get("from") == "doctor"])
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
        diagnosis_result = self.quality_agent.review_final_result(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        if diagnosis_result.get("_qc_issues"):
            logger.info(f"[质控] 诊疗方案修复/提示: {diagnosis_result['_qc_issues']}")

        logger.info(
            f"[诊断] 诊断结果: {json.dumps(diagnosis_result, ensure_ascii=False, indent=2)}"
        )

        # 提交诊疗方案
        submit_result = await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=diagnosis_result.get("diagnosis", []),
            treatment_plan=diagnosis_result.get("treatment_plan", ""),
            reasoning=diagnosis_result.get("reasoning", ""),
        )

        final_result = dict(diagnosis_result)
        if isinstance(submit_result, dict):
            for key, value in submit_result.items():
                if value is not None and value != "":
                    final_result[key] = value
        return self.quality_agent.review_final_result(
            final_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )

    # ============ 反思并保存经验 ============

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

        # 保存到记忆
        self.memory.save_case_experience(
            patient_id=patient_id,
            report=report,
            reflection=reflection,
            collected_info=collected_info,
            exam_results=exam_results,
        )

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
                if _n_cases > 0 and _n_cases % 5 == 0:
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
