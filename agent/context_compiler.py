"""Stage-specific LLM context compilation and audit."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping


class StageContextCompiler:
    """Compile authoritative runtime state into stage-specific LLM context."""

    VERSION = "stage_context_v1"

    DEFAULT_LIMITS = {
        "planning": {"chat_history": 10, "exam_results": 40},
        "thinking": {"chat_history": 12, "exam_results": 40},
        "diagnosis": {"chat_history": 16, "exam_results": 50},
        "planning_criticism": {"action_history": 10, "exam_results": 40},
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.compact_enabled = bool(self.config.get("compact_enabled", True))

    def compile(self, stage: str, **state: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "context": dict(state),
                "audit": self._audit(stage, state, state, [], ["compiler_disabled"]),
            }
        stage_name = str(stage or "unclassified")
        compiled = copy.deepcopy(dict(state))
        omitted: List[str] = []
        if self.compact_enabled:
            limits = self.DEFAULT_LIMITS.get(stage_name, {})
            self._trim_sequence(compiled, "chat_history", limits, omitted)
            self._trim_sequence(compiled, "action_history", limits, omitted)
            self._trim_mapping(compiled, "exam_results", limits, omitted)
        included = [
            key
            for key, value in compiled.items()
            if value not in (None, "", [], {})
        ]
        return {
            "context": compiled,
            "audit": self._audit(stage_name, state, compiled, included, omitted),
        }

    @staticmethod
    def _trim_sequence(
        payload: Dict[str, Any],
        key: str,
        limits: Mapping[str, int],
        omitted: List[str],
    ) -> None:
        limit = int(limits.get(key) or 0)
        value = payload.get(key)
        if limit <= 0 or not isinstance(value, list) or len(value) <= limit:
            return
        payload[key] = value[-limit:]
        omitted.append(f"{key}:{len(value) - limit}_old_items")

    @staticmethod
    def _trim_mapping(
        payload: Dict[str, Any],
        key: str,
        limits: Mapping[str, int],
        omitted: List[str],
    ) -> None:
        limit = int(limits.get(key) or 0)
        value = payload.get(key)
        if limit <= 0 or not isinstance(value, dict) or len(value) <= limit:
            return
        items = list(value.items())[-limit:]
        payload[key] = dict(items)
        omitted.append(f"{key}:{len(value) - limit}_old_items")

    def _audit(
        self,
        stage: str,
        source: Mapping[str, Any],
        compiled: Mapping[str, Any],
        included: List[str],
        omitted: List[str],
    ) -> Dict[str, Any]:
        source_text = _json_text(source)
        compiled_text = _json_text(compiled)
        return {
            "context_version": self.VERSION,
            "stage": stage,
            "source_case_version": str(source.get("case_version") or ""),
            "source_context_chars": len(source_text),
            "context_chars": len(compiled_text),
            "estimated_input_tokens": max(1, round(len(compiled_text) / 4)),
            "included_sections": list(included),
            "omitted_sections": list(omitted),
        }


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
