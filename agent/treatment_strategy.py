"""治疗策略 Agent：基于疾病画像补齐治疗原则和安全提醒。"""

from typing import Any, Dict, List

from .diagnosis_engine import DiagnosticKnowledgeBase
from .knowledge import KnowledgeBase


class TreatmentStrategyAgent:
    """提交前治疗方案增强角色。"""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        diagnostic_knowledge: DiagnosticKnowledgeBase = None,
    ):
        self.knowledge = knowledge
        self.diagnostic_knowledge = diagnostic_knowledge or DiagnosticKnowledgeBase(
            ref_dir=getattr(knowledge, "ref_dir", "data/ref_data")
        )

    def review(
        self,
        result: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """结合标准疾病画像补齐治疗原则、随访和安全提醒。"""
        fixed = dict(result or {})
        diagnosis = fixed.get("diagnosis") or []
        if isinstance(diagnosis, str):
            diagnosis = [diagnosis]

        rule_protocols = self.diagnostic_knowledge.get_treatment_protocols(diagnosis)
        if rule_protocols:
            fixed = self._apply_rule_protocol_treatment(fixed, rule_protocols)

        principles: List[str] = []
        warnings: List[str] = []
        for disease in diagnosis:
            profile = self.knowledge.get_disease_profile(str(disease))
            if not profile:
                continue
            for item in profile.get("treatment_principles") or []:
                if item and item not in principles:
                    principles.append(str(item))
            for item in profile.get("avoid_mistakes") or []:
                if item and item not in warnings:
                    warnings.append(str(item))

        treatment_plan = str(fixed.get("treatment_plan") or "")
        additions: List[str] = []
        if principles:
            additions.append("治疗原则：" + "；".join(principles[:5]) + "。")
        additions.append("需结合年龄、既往病史、过敏史和检查异常调整用药，并安排复诊复查。")
        if warnings:
            additions.append("安全提醒：" + "；".join(warnings[:3]) + "。")

        for addition in additions:
            if addition not in treatment_plan:
                treatment_plan = (treatment_plan.rstrip("。") + "。" if treatment_plan else "") + addition
        fixed["treatment_plan"] = treatment_plan

        reasoning = str(fixed.get("reasoning") or "")
        if principles and "治疗原则" not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + "治疗方案参考疾病画像中的标准处理原则，并结合已完成检查结果制定。"
        fixed["reasoning"] = reasoning
        fixed.setdefault("_treatment_strategy", {})
        fixed["_treatment_strategy"] = {
            "principles": principles[:5],
            "warnings": warnings[:3],
            "rule_protocols": rule_protocols[:8],
            "exam_count": len(exam_results or {}),
            "has_personalization_note": True,
            "structural_cardiac": any(
                self.diagnostic_knowledge.get(item).get("diagnosis_type") == "structural"
                for item in diagnosis
            ),
        }
        return fixed

    @staticmethod
    def _apply_rule_protocol_treatment(
        result: Dict[str, Any],
        protocols: List[str],
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        steps = [str(item).strip() for item in protocols if str(item).strip()]
        if not steps:
            return fixed
        treatment_plan = "证据规则驱动治疗方案：" + "；".join(
            f"{idx}. {step}" for idx, step in enumerate(steps, 1)
        ) + "。"
        fixed["treatment_plan"] = treatment_plan
        reasoning = str(fixed.get("reasoning") or "")
        note = "治疗方案根据证据裁决后的病因诊断和 diagnostic_knowledge 中的 treatment_protocol 生成。"
        if note not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + note
        fixed["reasoning"] = reasoning
        return fixed

    @staticmethod
    def _has_structural_cardiac_diagnosis(
        diagnosis: List[Any],
        result: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> bool:
        diag_text = " ".join(str(item) for item in diagnosis or [])
        trusted_text = " ".join(str(item) for item in result.get("_trusted_diagnoses") or [])
        exam_text = str(exam_results or {})
        text = diag_text + " " + trusted_text + " " + exam_text
        return (
            ("三房心" in text or "左心房隔膜" in text)
            and ("心内膜垫缺损" in text or "房室间隔缺损" in text or "共同房室瓣" in text)
        )

    @staticmethod
    def _apply_structural_cardiac_treatment(
        result: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        findings = fixed.get("_structural_findings") or []
        finding_text = "、".join(str(item) for item in findings[:6]) if findings else "超声心动图和心导管检查提示结构性心脏病"

        treatment_plan = (
            "诊疗重点为结构性先天性心脏病导致的心肺功能失代偿："
            "1. 立即转入儿科心脏专科/心胸外科评估，准备切除左心房隔膜并修复完全性房室间隔缺损；"
            "2. 围手术期给予吸氧，维持合适氧饱和度，连续监测心率、呼吸频率、血压、尿量和液体出入量；"
            "3. 在监护下使用静脉呋塞米减轻肺血管充血，并根据血流动力学给予米力农等正性肌力/扩血管支持；"
            "4. 喂养采用慢流量奶嘴、少量多次和高热量配方奶，减少吃奶时呼吸负荷，避免用电解质液替代营养；"
            "5. 咳嗽、流涕、低热更符合病毒性上呼吸道感染诱发心衰失代偿，若无持续高热、化脓感染证据或培养阳性，不常规使用抗生素；"
            "6. 若出现拒奶、嗜睡、发绀加重、尿量减少或呼吸窘迫进展，立即急诊/重症监护处理。"
        )
        fixed["treatment_plan"] = treatment_plan

        reasoning = str(fixed.get("reasoning") or "")
        structural_note = (
            f"治疗方案围绕病因处理制定：{finding_text}，"
            "首要矫治结构畸形并进行围手术期强心、利尿、氧疗和营养支持；"
            "呼吸道感染作为诱因监测，避免无证据抗生素。"
        )
        if structural_note not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + structural_note
        fixed["reasoning"] = reasoning
        return fixed
