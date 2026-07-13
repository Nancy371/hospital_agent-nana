"""治疗策略 Agent：基于疾病画像补齐治疗原则和安全提醒。"""

from typing import Any, Dict, List

from .knowledge import KnowledgeBase


class TreatmentStrategyAgent:
    """提交前治疗方案增强角色。"""

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge

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
            "exam_count": len(exam_results or {}),
            "has_personalization_note": True,
        }
        return fixed
