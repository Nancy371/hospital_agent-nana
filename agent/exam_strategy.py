"""检查策略 Agent：结合疾病画像补齐必查检查并过滤无效检查。"""

from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase


class ExamStrategyAgent:
    """轻量检查策略角色，不调用外部服务，只做本地规则增强。"""

    def __init__(self, knowledge: KnowledgeBase, max_new_items: int = 5):
        self.knowledge = knowledge
        self.max_new_items = max_new_items

    def recommend(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
        existing_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """返回本轮建议检查项。

        优先保留疾病画像中的必查检查，其次使用 LLM 提议的有效检查项。
        """
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        existing_results = existing_results or {}
        existing_valid, _ = self.knowledge.normalize_examinations(list(existing_results.keys()))
        existing_set = set(existing_results.keys()) | set(existing_valid)

        proposed_valid, invalid_items = self.knowledge.normalize_examinations(proposed_items or [])
        required_items = self.knowledge.get_required_exams(
            candidate_diseases=candidate_diseases or [],
            symptoms=symptoms,
            include_optional=False,
        )

        merged: List[str] = []
        for item in required_items + proposed_valid:
            if item and item not in existing_set and item not in merged:
                merged.append(item)

        return {
            "items": merged[: self.max_new_items],
            "required_items": required_items,
            "added_required": [item for item in required_items if item not in proposed_valid],
            "invalid_items": invalid_items,
            "clinical_context": self.knowledge.build_clinical_context(
                symptoms=symptoms,
                candidate_diseases=candidate_diseases,
            ),
        }
