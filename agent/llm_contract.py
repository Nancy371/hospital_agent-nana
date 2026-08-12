"""LLM JSON contract validation and bounded repair helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class LLMContractSchema:
    purpose: str
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class LLMContractValidation:
    purpose: str
    applicable: bool
    schema_success: Optional[bool]
    semantic_success: Optional[bool]
    missing_fields: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        if not self.applicable:
            return True
        return bool(self.schema_success) and self.semantic_success is not False

    def to_audit(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "applicable": self.applicable,
            "schema_success": self.schema_success,
            "semantic_success": self.semantic_success,
            "missing_fields": list(self.missing_fields),
            "type_errors": list(self.type_errors),
            "semantic_errors": list(self.semantic_errors),
            "accepted": self.accepted,
        }


class LLMContractExecutor:
    """Validate structured LLM outputs without making clinical decisions."""

    def __init__(self, *, enabled: bool = True, repair_enabled: bool = True):
        self.enabled = bool(enabled)
        self.repair_enabled = bool(repair_enabled)
        self.schemas: Dict[str, LLMContractSchema] = {
            "planning": LLMContractSchema(
                purpose="planning",
                required_fields=["strategy"],
                field_types={"strategy": (dict,)},
            ),
            "thinking": LLMContractSchema(
                purpose="thinking",
                required_fields=["differential_diagnosis"],
                field_types={"differential_diagnosis": (list,)},
            ),
            "diagnosis": LLMContractSchema(
                purpose="diagnosis",
                required_fields=["diagnosis"],
                field_types={"diagnosis": (list, str)},
            ),
        }

    def schema_for(self, purpose: str) -> Optional[LLMContractSchema]:
        if not self.enabled:
            return None
        return self.schemas.get(str(purpose or ""))

    def validate(self, value: Any, purpose: str) -> LLMContractValidation:
        schema = self.schema_for(purpose)
        if schema is None:
            return LLMContractValidation(
                purpose=str(purpose or "unclassified"),
                applicable=False,
                schema_success=None,
                semantic_success=None,
            )
        if not isinstance(value, dict):
            return LLMContractValidation(
                purpose=schema.purpose,
                applicable=True,
                schema_success=False,
                semantic_success=None,
                type_errors=["root_not_object"],
            )
        if set(value.keys()) == {"raw_response"} and isinstance(value.get("raw_response"), str):
            return LLMContractValidation(
                purpose=schema.purpose,
                applicable=True,
                schema_success=False,
                semantic_success=None,
                type_errors=["unparsed_raw_response"],
            )
        missing = [
            field
            for field in schema.required_fields
            if field not in value
        ]
        type_errors = []
        for field, expected in schema.field_types.items():
            if field in value and not isinstance(value.get(field), expected):
                type_errors.append(f"{field}:expected_{_type_names(expected)}")
        semantic_errors = self._semantic_errors(value, schema)
        schema_success = not missing and not type_errors
        semantic_success: Optional[bool] = None
        if schema_success:
            semantic_success = not semantic_errors
        return LLMContractValidation(
            purpose=schema.purpose,
            applicable=True,
            schema_success=schema_success,
            semantic_success=semantic_success,
            missing_fields=missing,
            type_errors=type_errors,
            semantic_errors=semantic_errors,
        )

    def should_repair(self, validation: LLMContractValidation) -> bool:
        return bool(
            self.enabled
            and self.repair_enabled
            and validation.applicable
            and not validation.accepted
        )

    def build_repair_messages(
        self,
        *,
        original_messages: List[Dict[str, str]],
        previous_value: Any,
        validation: LLMContractValidation,
    ) -> List[Dict[str, str]]:
        schema = self.schema_for(validation.purpose)
        required_fields = schema.required_fields if schema else []
        previous_payload = _safe_json(previous_value)
        return [
            {
                "role": "system",
                "content": (
                    "You repair a previous structured JSON response so it matches the "
                    "requested schema. Return only one JSON object. Do not redo clinical "
                    "reasoning, do not add prose, and do not invent evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "purpose": validation.purpose,
                        "required_fields": required_fields,
                        "validation_errors": validation.to_audit(),
                        "previous_response": previous_payload,
                        "instruction": (
                            "Correct only the representation. Preserve the prior answer's "
                            "clinical content when possible."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _semantic_errors(
        value: Mapping[str, Any],
        schema: LLMContractSchema,
    ) -> List[str]:
        errors: List[str] = []
        if schema.purpose == "diagnosis":
            diagnosis = value.get("diagnosis")
            if isinstance(diagnosis, list) and not [
                item for item in diagnosis if str(item).strip()
            ]:
                errors.append("diagnosis_empty")
            elif isinstance(diagnosis, str) and not diagnosis.strip():
                errors.append("diagnosis_empty")
        return errors


def _type_names(types: tuple) -> str:
    return "_or_".join(getattr(item, "__name__", str(item)) for item in types)


def _safe_json(value: Any, *, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text
