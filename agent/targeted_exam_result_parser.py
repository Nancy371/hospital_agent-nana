"""Gap-aware interpretation of returned examination results."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import Observation


POSITIVE = "positive"
NEGATIVE = "negative"
INCONCLUSIVE = "inconclusive"
UNRESOLVED = "unresolved"
UNBOUND = "unbound"

FULL = "full"
CONDITIONAL = "conditional"
PARTIAL = "partial"
NON_CLOSING = "non_closing"
UNSUPPORTED = "unsupported"


@dataclass
class ExamResultIntentBinding:
    binding_id: str
    order_id: str
    requested_exam: str
    resolved_exam: str = ""
    actual_result_exam: str = ""
    target_gap_ids: List[str] = field(default_factory=list)
    target_claims: List[str] = field(default_factory=list)
    target_candidate: str = ""
    entity_id: str = ""
    parser_profile: str = ""
    planned_resolution_type: str = ""
    planned_closure_level: str = ""
    actual_resolution_type: str = ""
    actual_closure_level: str = ""
    source_gap_value: float = 0.0
    source_decision_version: int = 0
    source_evidence_version: int = 0
    execution_status: str = "planned"
    binding_status: str = "bound"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, value: Any) -> Optional["ExamResultIntentBinding"]:
        if isinstance(value, ExamResultIntentBinding):
            return value
        if not isinstance(value, dict):
            return None
        requested = str(value.get("requested_exam") or value.get("exam") or "").strip()
        order_id = str(value.get("order_id") or "").strip()
        binding_id = str(value.get("binding_id") or "").strip()
        if not requested and not order_id:
            return None
        return cls(
            binding_id=binding_id or _stable_id("binding", order_id or requested),
            order_id=order_id or _stable_id("order", requested),
            requested_exam=requested,
            resolved_exam=str(value.get("resolved_exam") or requested),
            actual_result_exam=str(value.get("actual_result_exam") or ""),
            target_gap_ids=_text_list(
                value.get("target_gap_ids") or value.get("target_gaps") or []
            ),
            target_claims=_text_list(
                value.get("target_claims")
                or value.get("target_findings")
                or value.get("target_evidence")
                or []
            ),
            target_candidate=str(value.get("target_candidate") or ""),
            entity_id=str(value.get("entity_id") or ""),
            parser_profile=str(value.get("parser_profile") or ""),
            planned_resolution_type=str(value.get("planned_resolution_type") or value.get("resolution_type") or ""),
            planned_closure_level=str(value.get("planned_closure_level") or value.get("closure_level") or ""),
            actual_resolution_type=str(value.get("actual_resolution_type") or ""),
            actual_closure_level=str(value.get("actual_closure_level") or ""),
            source_gap_value=_float(value.get("source_gap_value"), 0.0),
            source_decision_version=_int(value.get("source_decision_version"), 0),
            source_evidence_version=_int(value.get("source_evidence_version"), 0),
            execution_status=str(value.get("execution_status") or "planned"),
            binding_status=str(value.get("binding_status") or "bound"),
        )


@dataclass
class TargetedExamParseResult:
    binding_id: str
    order_id: str
    target_gap_ids: List[str]
    entity_id: str
    parser_profile: str
    status: str = UNRESOLVED
    observations: List[Observation] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    unmatched_target_claims: List[str] = field(default_factory=list)
    source_text: str = ""
    actual_closure_level: str = ""
    gap_closure_assessment: str = "inconclusive"
    binding_status: str = "bound"
    actual_result_exam: str = ""
    execution_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["observations"] = [item.to_dict() for item in self.observations]
        return payload


class TargetedExamResultParser:
    """Parse returned results according to the gap that caused the order."""

    def parse(
        self,
        raw_exam_result: Any,
        intent_binding: Any,
    ) -> TargetedExamParseResult:
        binding = ExamResultIntentBinding.from_any(intent_binding)
        if binding is None:
            return TargetedExamParseResult(
                binding_id="",
                order_id="",
                target_gap_ids=[],
                entity_id="",
                parser_profile="",
                status=UNBOUND,
                binding_status="unbound",
                gap_closure_assessment="unbound",
            )

        binding = self._with_actual_capability(binding)
        base = TargetedExamParseResult(
            binding_id=binding.binding_id,
            order_id=binding.order_id,
            target_gap_ids=list(binding.target_gap_ids),
            entity_id=binding.entity_id,
            parser_profile=binding.parser_profile,
            source_text=_flatten_result_text(raw_exam_result),
            actual_closure_level=binding.actual_closure_level,
            binding_status=binding.binding_status,
            actual_result_exam=binding.actual_result_exam,
            execution_status=binding.execution_status,
        )
        if not binding.target_gap_ids or not self._is_pavm_binding(binding):
            base.status = UNBOUND if not binding.target_gap_ids else UNRESOLVED
            base.gap_closure_assessment = "not_closable"
            return base
        return self._parse_pavm(raw_exam_result, binding, base)

    def parse_all(
        self,
        result_by_exam: Dict[str, Any],
        bindings: Sequence[Any],
    ) -> List[TargetedExamParseResult]:
        parsed: List[TargetedExamParseResult] = []
        for binding_value in bindings or []:
            binding = ExamResultIntentBinding.from_any(binding_value)
            if binding is None:
                continue
            exam_name = binding.actual_result_exam or binding.resolved_exam or binding.requested_exam
            raw = _lookup_result(result_by_exam, exam_name)
            if raw is None:
                continue
            parsed.append(self.parse(raw, binding))
        return parsed

    def _parse_pavm(
        self,
        raw_exam_result: Any,
        binding: ExamResultIntentBinding,
        result: TargetedExamParseResult,
    ) -> TargetedExamParseResult:
        text = result.source_text
        compact = _compact(text)
        level = binding.actual_closure_level or NON_CLOSING
        observations: List[Observation] = []
        matched: List[str] = []

        def add(
            finding: str,
            confidence: float,
            *,
            polarity: str = "positive",
            evidence_level: str = "observed_imaging",
            information_value: float = 0.9,
            rule_id: str = "",
        ) -> None:
            observations.append(
                Observation(
                    finding=finding,
                    source="targeted_exam_result_parser",
                    polarity=polarity,
                    confidence=confidence,
                    raw_text=text,
                    source_text=_support_span(text, rule_id) or text[:320],
                    field_path=f"exam_result.{binding.actual_result_exam or binding.resolved_exam or binding.requested_exam}",
                    evidence_level=evidence_level,
                    information_value=information_value,
                    source_exam=binding.actual_result_exam or binding.resolved_exam or binding.requested_exam,
                    order_id=binding.order_id,
                    target_gap_ids=list(binding.target_gap_ids),
                    entity_id=binding.entity_id,
                    verification_method="targeted_exam_result_parser",
                    parser_profile=binding.parser_profile,
                    gap_closure_assessment=result.gap_closure_assessment,
                )
            )
            if rule_id:
                matched.append(rule_id)

        direct = _has_any(
            compact,
            (
                "肺动静脉瘘",
                "肺动静脉畸形",
                "肺动静脉交通",
                "动静脉异常交通",
                "动静脉瘘样血管结构",
                "pavm",
                "pulmonaryarteriovenousmalformation",
            ),
        )
        feeding = _has_any(compact, ("供血肺动脉", "供血动脉", "feedingpulmonaryartery", "feedingartery"))
        draining = _has_any(compact, ("引流肺静脉", "早期引流肺静脉", "drainingpulmonaryvein", "drainingvein"))
        early_vein = _has_any(compact, ("早期肺静脉显影", "肺静脉早期显影", "earlypulmonaryvenousenhancement"))
        vascular_cluster = _has_any(
            compact,
            ("异常肺血管团", "异常血管团", "强化血管团", "迂曲血管", "血管性病变", "肺血管畸形"),
        )
        bubble_positive = _has_any(
            compact,
            ("微泡延迟进入左心", "微泡进入左心", "延迟显影", "右向左分流", "右至左分流", "肺内分流", "intrapulmonaryshunt"),
        )
        support_only = _has_any(compact, ("肺内结节", "肺结节", "血管影增粗", "圆形致密影", "边界清楚结节"))
        negative = _has_any(
            compact,
            (
                "未见肺动静脉",
                "未见动静脉异常交通",
                "未见肺血管畸形",
                "未见供血动脉",
                "未见引流静脉",
                "无右向左分流",
                "声学造影阴性",
                "未见微泡进入左心",
            ),
        )

        if negative and level in {FULL, CONDITIONAL}:
            result.status = NEGATIVE
            result.gap_closure_assessment = "negative_closed"
            for target in binding.target_claims or ["pulmonary_cta_positive"]:
                add(
                    str(target),
                    0.88,
                    polarity="negative",
                    information_value=0.88,
                    rule_id="pavm_effective_negative",
                )
            result.observations = _dedupe_observations(observations)
            result.matched_rules = list(dict.fromkeys(matched))
            return result

        vascular_exam_can_answer = level in {FULL, CONDITIONAL}
        if vascular_exam_can_answer:
            if direct:
                add("abnormal_pulmonary_av_connection_described", 0.97, information_value=0.98, rule_id="pavm_direct_diagnosis_phrase")
                add("pulmonary_avm_mechanism", 0.93, information_value=0.94, rule_id="pavm_direct_mechanism")
            if feeding:
                add("feeding_pulmonary_artery_present", 0.95, information_value=0.94, rule_id="pavm_feeding_artery")
            if draining:
                add("draining_pulmonary_vein_present", 0.95, information_value=0.94, rule_id="pavm_draining_vein")
            if early_vein:
                add("early_pulmonary_venous_enhancement", 0.94, information_value=0.93, rule_id="pavm_early_venous_enhancement")
            if vascular_cluster:
                add("abnormal_pulmonary_vascular_cluster", 0.88, information_value=0.82, rule_id="pavm_vascular_cluster")
        elif direct or feeding or draining or early_vein or vascular_cluster or support_only:
            add(
                "vascular_pulmonary_nodule_suspected",
                0.72,
                evidence_level="support",
                information_value=0.58,
                rule_id="pavm_non_closing_vascular_support",
            )
        if bubble_positive and "echo" in binding.parser_profile:
            add("delayed_bubbles_in_left_heart", 0.96, information_value=0.96, rule_id="pavm_bubble_delayed_left_heart")
            add("intrapulmonary_right_to_left_shunt_observed", 0.95, information_value=0.96, rule_id="pavm_bubble_shunt")
            add("bubble_echo_right_to_left_shunt", 0.96, evidence_level="diagnostic_pattern", information_value=0.97, rule_id="pavm_bubble_echo_positive")
            add("right_to_left_shunt", 0.92, evidence_level="diagnostic_pattern", information_value=0.96, rule_id="pavm_right_to_left_shunt_positive")
            add("pulmonary_vascular_shunt", 0.88, evidence_level="diagnostic_pattern", information_value=0.92, rule_id="pavm_pulmonary_shunt_positive")

        vascular_combo = direct or (feeding and draining) or (vascular_cluster and early_vein)
        if vascular_combo and level in {FULL, CONDITIONAL}:
            if "cta" in binding.parser_profile:
                add("pulmonary_cta_positive", 0.97, evidence_level="diagnostic_pattern", information_value=0.98, rule_id="pavm_cta_confirmed")
            elif "enhanced_ct" in binding.parser_profile:
                add("enhanced_ct_vascular_malformation", 0.96, evidence_level="diagnostic_pattern", information_value=0.97, rule_id="pavm_enhanced_ct_confirmed")
            else:
                add("enhanced_ct_vascular_malformation", 0.92, evidence_level="diagnostic_pattern", information_value=0.92, rule_id="pavm_vascular_confirmation")
            add("pulmonary_avm_imaging", 0.94, evidence_level="diagnostic_pattern", information_value=0.94, rule_id="pavm_imaging_positive")
            add("pulmonary_avm_mechanism", 0.9, evidence_level="diagnostic_pattern", information_value=0.94, rule_id="pavm_mechanism_positive")
            if draining or early_vein:
                add("right_to_left_shunt", 0.88, evidence_level="diagnostic_pattern", information_value=0.9, rule_id="pavm_ct_derived_shunt")
                add("pulmonary_vascular_shunt", 0.86, evidence_level="diagnostic_pattern", information_value=0.88, rule_id="pavm_ct_derived_pulmonary_shunt")

        if observations:
            if any(item.finding in {"pulmonary_cta_positive", "enhanced_ct_vascular_malformation", "bubble_echo_right_to_left_shunt"} for item in observations):
                result.status = POSITIVE
                result.gap_closure_assessment = "positive_closed"
            elif level in {PARTIAL, NON_CLOSING} or support_only:
                result.status = INCONCLUSIVE
                result.gap_closure_assessment = "partial"
            else:
                result.status = INCONCLUSIVE
                result.gap_closure_assessment = "inconclusive"
        elif support_only:
            result.status = INCONCLUSIVE
            result.gap_closure_assessment = "partial"
            add(
                "vascular_pulmonary_nodule_suspected",
                0.72,
                evidence_level="support",
                information_value=0.58,
                rule_id="pavm_generic_vascular_nodule_support",
            )
        else:
            result.status = UNRESOLVED
            result.gap_closure_assessment = "not_closed"

        for item in observations:
            item.gap_closure_assessment = result.gap_closure_assessment
        result.observations = _dedupe_observations(observations)
        result.matched_rules = list(dict.fromkeys(matched))
        result.unmatched_target_claims = [
            claim for claim in binding.target_claims or [] if claim not in {item.finding for item in result.observations}
        ]
        return result

    def _with_actual_capability(
        self,
        binding: ExamResultIntentBinding,
    ) -> ExamResultIntentBinding:
        actual = binding.actual_result_exam or binding.resolved_exam or binding.requested_exam
        profile, level, resolution = _capability_for_exam(actual, binding)
        binding.parser_profile = binding.parser_profile or profile
        binding.actual_closure_level = level
        binding.actual_resolution_type = resolution
        if not binding.planned_closure_level:
            binding.planned_closure_level = level
        if not binding.planned_resolution_type:
            binding.planned_resolution_type = resolution
        return binding

    @staticmethod
    def _is_pavm_binding(binding: ExamResultIntentBinding) -> bool:
        text = _compact(
            " ".join(
                [
                    binding.entity_id,
                    binding.target_candidate,
                    " ".join(binding.target_claims),
                    " ".join(binding.target_gap_ids),
                    binding.parser_profile,
                ]
            )
        )
        return any(
            marker in text
            for marker in (
                "d100055",
                "pavm",
                "肺动静脉瘘",
                "肺动静脉畸形",
                "righttoleftshunt",
                "pulmonaryvascular",
                "pulmonarycta",
            )
        )


def binding_from_authorization_detail(
    *,
    detail: Dict[str, Any],
    requested_exam: str,
    actual_result_exam: str = "",
    patient_id: str = "",
    stage: str = "",
    order_index: int = 0,
) -> ExamResultIntentBinding:
    order_id = _stable_id(
        "exam",
        patient_id,
        stage,
        str(order_index),
        requested_exam,
        actual_result_exam,
        ",".join(_text_list(detail.get("target_gaps") or [])),
    )
    target_candidates = _text_list(detail.get("target_candidates") or [])
    candidate = target_candidates[0] if target_candidates else ""
    entity_id = str(detail.get("entity_id") or "")
    binding = ExamResultIntentBinding(
        binding_id=_stable_id("binding", order_id),
        order_id=order_id,
        requested_exam=str(detail.get("requested_exam") or requested_exam),
        resolved_exam=str(detail.get("resolved_exam") or detail.get("exam") or requested_exam),
        actual_result_exam=actual_result_exam,
        target_gap_ids=_text_list(detail.get("target_gaps") or []),
        target_claims=_text_list(detail.get("target_claims") or detail.get("target_findings") or []),
        target_candidate=candidate,
        entity_id=entity_id,
        planned_resolution_type=str(detail.get("resolution_type") or ""),
        planned_closure_level=str(detail.get("closure_level") or ""),
        source_gap_value=_float(detail.get("source_gap_value"), 0.0),
        source_decision_version=_int(detail.get("source_decision_version"), 0),
        source_evidence_version=_int(detail.get("source_evidence_version"), 0),
        execution_status="result_received" if actual_result_exam else "ordered",
    )
    profile, level, resolution = _capability_for_exam(actual_result_exam or binding.resolved_exam, binding)
    binding.parser_profile = profile
    binding.actual_closure_level = level
    binding.actual_resolution_type = resolution
    if not binding.planned_closure_level:
        binding.planned_closure_level = level
    if not binding.planned_resolution_type:
        binding.planned_resolution_type = resolution
    return binding


def _capability_for_exam(
    exam_name: str,
    binding: ExamResultIntentBinding,
) -> Tuple[str, str, str]:
    compact = _compact(exam_name)
    if not TargetedExamResultParser._is_pavm_binding(binding):
        return "generic_exam_v1", NON_CLOSING, "non_closing"
    if any(marker in compact for marker in ("肺动脉cta", "肺血管cta", "肺动脉ct血管成像", "pulmonarycta", "ctangiography")):
        return "pavm_cta_v1", FULL, "equivalent"
    if any(marker in compact for marker in ("右心声学造影", "超声心动图右心声学造影", "bubbleecho", "bubblestudy")):
        return "pavm_bubble_echo_v1", FULL, "equivalent"
    if any(marker in compact for marker in ("胸部增强ct", "增强胸部ct", "chestcect", "contrastenhancedchestct")):
        return "pavm_enhanced_ct_v1", CONDITIONAL, "conditional"
    if "血管造影" in compact:
        return "pavm_angiography_v1", FULL, "equivalent"
    if any(marker in compact for marker in ("胸部ct", "chestct", "ct")):
        return "pavm_plain_ct_v1", PARTIAL, "partial"
    if "超声心动图" in compact or "心超" in compact:
        return "standard_echo_v1", NON_CLOSING, "non_closing"
    return "pavm_unresolved_exam_v1", UNSUPPORTED, "unresolved"


def _lookup_result(result_by_exam: Dict[str, Any], exam_name: str) -> Any:
    if exam_name in result_by_exam:
        return result_by_exam[exam_name]
    target = _compact(exam_name)
    for key, value in (result_by_exam or {}).items():
        if _compact(key) == target:
            return value
    return None


def _flatten_result_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            child = _flatten_result_text(item)
            if child:
                parts.append(f"{key}: {child}")
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_result_text(item) for item in value)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    except TypeError:
        return str(value)


def _support_span(text: str, marker: str) -> str:
    if not text:
        return ""
    if not marker:
        return text[:320]
    # Keep a concise source snippet around any Chinese/English clinical token.
    candidate_terms = [
        "肺动静脉",
        "供血",
        "引流",
        "早期肺静脉",
        "微泡",
        "右向左分流",
        "血管畸形",
        "异常血管",
    ]
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in candidate_terms if lowered.find(term.lower()) >= 0]
    if not positions:
        return text[:320]
    start = max(0, min(positions) - 50)
    end = min(len(text), min(positions) + 220)
    return text[start:end].strip()


def _dedupe_observations(values: Sequence[Observation]) -> List[Observation]:
    best: Dict[Tuple[str, str], Observation] = {}
    for item in values or []:
        key = (str(item.finding or ""), str(item.polarity or "positive"))
        if not key[0]:
            continue
        current = best.get(key)
        score = float(item.confidence or 0.0) + float(item.information_value or 0.0)
        current_score = (
            float(current.confidence or 0.0) + float(current.information_value or 0.0)
            if current
            else -1.0
        )
        if current is None or score > current_score:
            best[key] = item
    return list(best.values())


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(_compact(term) in text for term in terms if str(term or "").strip())


def _compact(value: Any) -> str:
    return re.sub(r"[\s_\-（）()［\]\[\]、，,。：:；;]+", "", str(value or "").lower())


def _text_list(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
