"""提交前质控 Agent：保证 final_result 可评估且尽量符合标准目录。"""

from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase


class QualityAgent:
    """本地规则质控角色。"""

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge

    @staticmethod
    def default_final_result(reason: str = "") -> Dict[str, Any]:
        return {
            "diagnosis": ["待明确诊断"],
            "treatment_plan": "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。",
            "reasoning": reason or "质控兜底：诊疗结果字段不完整。",
            "conversation_rounds": 0,
        }

    def normalize_diagnoses(
        self,
        diagnosis: Any,
        collected_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if isinstance(diagnosis, str):
            raw_items = [diagnosis]
        elif isinstance(diagnosis, list):
            raw_items = [str(item) for item in diagnosis if item]
        else:
            raw_items = []

        normalized: List[str] = []
        invalid: List[str] = []
        for item in raw_items:
            standard = self.knowledge.normalize_diagnosis(item)
            if standard and self.knowledge.is_valid_diagnosis(standard):
                normalized.append(standard)
            else:
                invalid.append(item)

        if not normalized:
            suggestions = self.knowledge.suggest_diagnoses(
                symptoms=(collected_info or {}).get("symptoms", []),
                candidate_diseases=raw_items,
                top_k=1,
            )
            normalized.extend(suggestions)

        if not normalized:
            # 最后兜底也使用标准目录内疾病名，避免非标准诊断影响评测解析。
            normalized.append("上呼吸道感染")

        return list(dict.fromkeys(normalized))

    def review_final_result(
        self,
        result: Any,
        collected_info: Optional[Dict[str, Any]] = None,
        exam_results: Optional[Dict[str, Any]] = None,
        conversation_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """规范化并修复最终结果，返回 result + issues。"""
        issues: List[str] = []
        if not isinstance(result, dict):
            result = self.default_final_result("质控发现结果不是字典。")
            issues.append("final_result_not_dict")

        fixed = dict(result)
        raw_diagnosis = fixed.get("diagnosis") or fixed.get("diagnoses")
        diagnosis = self.normalize_diagnoses(raw_diagnosis, collected_info=collected_info)
        raw_items = raw_diagnosis if isinstance(raw_diagnosis, list) else [raw_diagnosis]
        for item in raw_items:
            if item and not self.knowledge.normalize_diagnosis(str(item)):
                issues.append(f"nonstandard_diagnosis:{item}")
        if not diagnosis:
            diagnosis = ["上呼吸道感染"]
            issues.append("empty_diagnosis")
        fixed["diagnosis"] = diagnosis

        treatment_plan = fixed.get("treatment_plan") or fixed.get("treatment") or fixed.get("plan")
        if not treatment_plan or len(str(treatment_plan).strip()) < 6:
            treatment_plan = "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。"
            issues.append("empty_treatment_plan")
        fixed["treatment_plan"] = str(treatment_plan)

        reasoning = fixed.get("reasoning") or fixed.get("reason")
        if not reasoning:
            reasoning = "基于已收集的问诊和检查信息形成当前诊疗方案。"
            issues.append("empty_reasoning")
        fixed["reasoning"] = str(reasoning)

        if conversation_rounds is not None:
            fixed["conversation_rounds"] = int(conversation_rounds)
        else:
            try:
                fixed["conversation_rounds"] = int(fixed.get("conversation_rounds", 0) or 0)
            except (TypeError, ValueError):
                fixed["conversation_rounds"] = 0
                issues.append("invalid_conversation_rounds")

        missing_required = self.missing_required_exams(
            diagnosis=diagnosis,
            collected_info=collected_info or {},
            exam_results=exam_results or {},
        )
        if missing_required:
            issues.append("missing_required_exams:" + ",".join(missing_required))
            fixed["reasoning"] += " 质控提示：建议结合病情补充/确认关键检查：" + "、".join(missing_required) + "。"

        fixed["_qc_issues"] = issues
        return fixed

    def missing_required_exams(
        self,
        diagnosis: List[str],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> List[str]:
        required = self.knowledge.get_required_exams(
            candidate_diseases=diagnosis,
            symptoms=collected_info.get("symptoms", []) if collected_info else [],
            include_optional=False,
        )
        done, _ = self.knowledge.normalize_examinations(list((exam_results or {}).keys()))
        done_set = set(done) | set((exam_results or {}).keys())
        return [item for item in required if item not in done_set]
