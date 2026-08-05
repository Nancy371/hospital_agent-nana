"""Clinical explanation comparator for primary arbitration.

The comparator is intentionally narrow: it compares two already-admitted
candidates on verified evidence, diagnostic/bridge patterns, anchor validity,
and actionable gaps. It never mutates candidates and never authorizes
submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from .clinical_pattern_bridge import CROSS_SYSTEM_SCOPE, has_active_bridge_protection
from .diagnosis_eligibility import DEFERRED, EXCLUDED, PRIMARY_ELIGIBLE


ANCHOR_SATISFIED = "AnchorSatisfied"
PATTERN_SUPPORTED_BUT_UNCONFIRMED = "PatternSupportedButUnconfirmed"
NO_VALID_ANCHOR = "NoValidAnchor"
HARD_BLOCKED = "HardBlocked"

KEEP_CURRENT_PRIMARY = "KEEP_CURRENT_PRIMARY"
SWITCH_PRIMARY = "SWITCH_PRIMARY"
UNLOCK_AND_DEFER = "UNLOCK_AND_DEFER"
KEEP_CURRENT_AND_DEFER_CONTENDER = "KEEP_CURRENT_AND_DEFER_CONTENDER"
REJECT_CONTENDER = "REJECT_CONTENDER"
NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_DIFFERENCE"

_ANCHOR_RANK = {
    HARD_BLOCKED: -2,
    NO_VALID_ANCHOR: 0,
    PATTERN_SUPPORTED_BUT_UNCONFIRMED: 1,
    ANCHOR_SATISFIED: 2,
}

_BROAD_EVIDENCE = {
    "acute_course",
    "chronic_course",
    "cough",
    "dyspnea",
    "fatigue",
    "fever",
    "pain",
    "rash",
    "symptom:signal",
    "weakness",
}


@dataclass
class CandidateClinicalAnalysis:
    diagnosis: str
    entity_id: str = ""
    anchor_status: str = NO_VALID_ANCHOR
    eligibility_status: str = ""
    explained_core_evidence: List[str] = field(default_factory=list)
    explained_high_value_evidence: List[str] = field(default_factory=list)
    unexplained_high_value_evidence: List[str] = field(default_factory=list)
    contradicted_evidence: List[str] = field(default_factory=list)
    matched_diagnostic_patterns: List[str] = field(default_factory=list)
    matched_bridge_patterns: List[str] = field(default_factory=list)
    actionable_gaps: List[str] = field(default_factory=list)
    judge_score: float = 0.0
    candidate_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "entity_id": self.entity_id,
            "anchor_status": self.anchor_status,
            "eligibility_status": self.eligibility_status,
            "explained_core_evidence": list(self.explained_core_evidence),
            "explained_high_value_evidence": list(self.explained_high_value_evidence),
            "unexplained_high_value_evidence": list(self.unexplained_high_value_evidence),
            "contradicted_evidence": list(self.contradicted_evidence),
            "matched_diagnostic_patterns": list(self.matched_diagnostic_patterns),
            "matched_bridge_patterns": list(self.matched_bridge_patterns),
            "actionable_gaps": list(self.actionable_gaps),
            "judge_score": round(float(self.judge_score or 0.0), 4),
            "candidate_score": round(float(self.candidate_score or 0.0), 4),
        }


class ClinicalReasoningComparator:
    """Compare clinical explanatory adequacy before locking a primary."""

    def compare(
        self,
        current_primary: Any,
        contender: Any,
        *,
        judge_score_current: float = 0.0,
        judge_score_contender: float = 0.0,
        high_value_evidence: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        primary_analysis = self.analyze(
            current_primary,
            peer=contender,
            high_value_evidence=high_value_evidence,
            judge_score=judge_score_current,
        )
        contender_analysis = self.analyze(
            contender,
            peer=current_primary,
            high_value_evidence=high_value_evidence,
            judge_score=judge_score_contender,
        )
        reason_codes: List[str] = []

        if primary_analysis.anchor_status == HARD_BLOCKED:
            reason_codes.append("CURRENT_PRIMARY_HARD_BLOCKED")
        if contender_analysis.anchor_status == HARD_BLOCKED:
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=current_primary,
                action=REJECT_CONTENDER,
                reason_codes=["CONTENDER_HARD_BLOCKED"],
            )

        primary_anchor_rank = _ANCHOR_RANK.get(primary_analysis.anchor_status, 0)
        contender_anchor_rank = _ANCHOR_RANK.get(contender_analysis.anchor_status, 0)
        primary_explained = len(primary_analysis.explained_high_value_evidence)
        contender_explained = len(contender_analysis.explained_high_value_evidence)
        primary_residual = len(primary_analysis.unexplained_high_value_evidence)
        contender_residual = len(contender_analysis.unexplained_high_value_evidence)
        contender_has_syndrome = bool(
            contender_analysis.matched_bridge_patterns
            or contender_analysis.matched_diagnostic_patterns
        )

        if primary_analysis.anchor_status == NO_VALID_ANCHOR:
            reason_codes.append("CURRENT_PRIMARY_HAS_NO_VALID_ANCHOR")
            if contender_explained > primary_explained or contender_residual < primary_residual:
                reason_codes.append("CONTENDER_EXPLAINS_HIGH_VALUE_EVIDENCE_BETTER")
                if contender_has_syndrome:
                    reason_codes.append("CONTENDER_HAS_VERIFIED_PATTERN_OR_BRIDGE")
                if contender_analysis.anchor_status == ANCHOR_SATISFIED:
                    return self._record(
                        current_primary,
                        contender,
                        primary_analysis,
                        contender_analysis,
                        preferred=contender,
                        action=SWITCH_PRIMARY,
                        reason_codes=reason_codes,
                    )
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=UNLOCK_AND_DEFER,
                    reason_codes=reason_codes
                    + ["CONTENDER_REQUIRES_CONFIRMATORY_GAP"],
                )

        if (
            contender_has_syndrome
            and contender_explained >= primary_explained + 2
            and contender_residual + 1 <= primary_residual
            and contender_anchor_rank >= primary_anchor_rank
        ):
            reason_codes.extend(
                [
                    "CONTENDER_EXPLAINS_HIGH_VALUE_EVIDENCE_BETTER",
                    "SCORE_DOWNGRADED_TO_TIE_BREAKER",
                ]
            )
            if contender_analysis.anchor_status == ANCHOR_SATISFIED:
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=SWITCH_PRIMARY,
                    reason_codes=reason_codes,
                )
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=contender,
                action=UNLOCK_AND_DEFER,
                reason_codes=reason_codes + ["CONTENDER_REQUIRES_CONFIRMATORY_GAP"],
            )

        if primary_analysis.anchor_status == ANCHOR_SATISFIED and contender_has_syndrome:
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=current_primary,
                action=KEEP_CURRENT_AND_DEFER_CONTENDER,
                reason_codes=["CURRENT_PRIMARY_HAS_VALID_ANCHOR"],
            )

        return self._record(
            current_primary,
            contender,
            primary_analysis,
            contender_analysis,
            preferred=current_primary,
            action=NO_MATERIAL_DIFFERENCE,
            reason_codes=["SCORE_ONLY_TIE_BREAKER_NOT_INVOKED"],
        )

    def analyze(
        self,
        candidate: Any,
        *,
        peer: Any = None,
        high_value_evidence: Sequence[str] | None = None,
        judge_score: float = 0.0,
    ) -> CandidateClinicalAnalysis:
        if not candidate:
            return CandidateClinicalAnalysis(diagnosis="")
        matched_diagnostic = self._matched_diagnostic_patterns(candidate)
        matched_bridge = self._matched_bridge_patterns(candidate)
        high_value = set(high_value_evidence or self._high_value_universe(candidate, peer))
        explained = self._explained_findings(candidate)
        explained_high = sorted(explained & high_value)
        residual = self._residual_findings(candidate)
        residual_high = sorted(residual & high_value)
        if matched_bridge:
            explained_high.extend(item for item in matched_bridge if item not in explained_high)
        if matched_diagnostic:
            explained_high.extend(item for item in matched_diagnostic if item not in explained_high)
        return CandidateClinicalAnalysis(
            diagnosis=str(getattr(candidate, "diagnosis", "") or ""),
            entity_id=str(getattr(candidate, "entity_id", "") or ""),
            anchor_status=self.anchor_status(candidate),
            eligibility_status=str(getattr(candidate, "eligibility_status", "") or ""),
            explained_core_evidence=sorted(explained - _BROAD_EVIDENCE)[:12],
            explained_high_value_evidence=list(dict.fromkeys(explained_high))[:12],
            unexplained_high_value_evidence=residual_high[:12],
            contradicted_evidence=self._contradicted_findings(candidate)[:12],
            matched_diagnostic_patterns=matched_diagnostic,
            matched_bridge_patterns=matched_bridge,
            actionable_gaps=self._actionable_gaps(candidate)[:12],
            judge_score=judge_score,
            candidate_score=float(getattr(candidate, "score", 0.0) or 0.0),
        )

    def anchor_status(self, candidate: Any) -> str:
        if not candidate:
            return NO_VALID_ANCHOR
        if bool(getattr(candidate, "hard_contradiction", False)):
            return HARD_BLOCKED
        explicit = str(getattr(candidate, "eligibility_anchor_status", "") or "")
        if explicit:
            return explicit
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status == EXCLUDED:
            return HARD_BLOCKED
        if self._matched_diagnostic_patterns(candidate) and status == PRIMARY_ELIGIBLE:
            return ANCHOR_SATISFIED
        if status == PRIMARY_ELIGIBLE and bool(getattr(candidate, "required_met", False)):
            return ANCHOR_SATISFIED
        if status == DEFERRED and (
            self._matched_bridge_patterns(candidate)
            or getattr(candidate, "required_gaps", None)
        ):
            return PATTERN_SUPPORTED_BUT_UNCONFIRMED
        if self._matched_bridge_patterns(candidate):
            return PATTERN_SUPPORTED_BUT_UNCONFIRMED
        return NO_VALID_ANCHOR

    def material_contender(self, candidate: Any, current_primary: Any = None) -> bool:
        if not candidate or bool(getattr(candidate, "hard_contradiction", False)):
            return False
        if str(getattr(candidate, "eligibility_status", "") or "") == EXCLUDED:
            return False
        if has_active_bridge_protection(candidate, CROSS_SYSTEM_SCOPE):
            return True
        if self._matched_bridge_patterns(candidate):
            return True
        if self._matched_diagnostic_patterns(candidate):
            return True
        if self.anchor_status(current_primary) != NO_VALID_ANCHOR:
            return False
        high_value = self._high_value_universe(candidate, current_primary)
        return len(self._explained_findings(candidate) & high_value) >= 2

    def pair_high_value_evidence(self, left: Any, right: Any) -> List[str]:
        return sorted(self._high_value_universe(left, right))

    @staticmethod
    def _record(
        current_primary: Any,
        contender: Any,
        primary_analysis: CandidateClinicalAnalysis,
        contender_analysis: CandidateClinicalAnalysis,
        *,
        preferred: Any,
        action: str,
        reason_codes: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "comparison_id": (
                "CRC-"
                + str(getattr(current_primary, "entity_id", "") or getattr(current_primary, "diagnosis", "primary"))
                + "-"
                + str(getattr(contender, "entity_id", "") or getattr(contender, "diagnosis", "contender"))
            ),
            "candidate_a": str(getattr(current_primary, "diagnosis", "") or ""),
            "candidate_b": str(getattr(contender, "diagnosis", "") or ""),
            "candidate_a_analysis": primary_analysis.to_dict(),
            "candidate_b_analysis": contender_analysis.to_dict(),
            "preferred_candidate": str(getattr(preferred, "diagnosis", "") or ""),
            "recommended_action": action,
            "decision_reason_codes": list(dict.fromkeys(reason_codes)),
        }

    def _high_value_universe(self, left: Any, right: Any = None) -> Set[str]:
        values: Set[str] = set()
        for candidate in (left, right):
            if not candidate:
                continue
            values.update(self._texts(getattr(candidate, "core_matched_evidence", []) or []))
            values.update(self._texts(getattr(candidate, "diagnostic_matched_evidence", []) or []))
            values.update(self._texts(getattr(candidate, "unexplained_core_evidence", []) or []))
            for pattern in getattr(candidate, "clinical_pattern_matches", []) or []:
                if isinstance(pattern, dict) and str(pattern.get("verification_status") or "") == "verified":
                    values.update(self._texts(pattern.get("supporting_findings") or []))
                    pattern_id = str(pattern.get("pattern_id") or "")
                    if pattern_id:
                        values.add(pattern_id)
            for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
                if isinstance(assertion, dict):
                    pattern_id = str(assertion.get("canonical_pattern") or "")
                    if pattern_id:
                        values.add(pattern_id)
        return {item for item in values if item and item not in _BROAD_EVIDENCE}

    def _explained_findings(self, candidate: Any) -> Set[str]:
        values: Set[str] = set()
        values.update(self._texts(getattr(candidate, "core_matched_evidence", []) or []))
        values.update(self._texts(getattr(candidate, "diagnostic_matched_evidence", []) or []))
        values.update(
            item
            for item in self._texts(getattr(candidate, "matched_evidence", []) or [])
            if item not in _BROAD_EVIDENCE and not item.startswith("field:")
        )
        for pattern in getattr(candidate, "clinical_pattern_matches", []) or []:
            if isinstance(pattern, dict) and str(pattern.get("verification_status") or "") == "verified":
                values.update(self._texts(pattern.get("supporting_findings") or []))
                pattern_id = str(pattern.get("pattern_id") or "")
                if pattern_id:
                    values.add(pattern_id)
        for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
            if isinstance(assertion, dict):
                pattern_id = str(assertion.get("canonical_pattern") or "")
                if pattern_id:
                    values.add(pattern_id)
        return values

    def _residual_findings(self, candidate: Any) -> Set[str]:
        values = set(self._texts(getattr(candidate, "residual_evidence", []) or []))
        values.update(self._texts(getattr(candidate, "unexplained_core_evidence", []) or []))
        return values

    def _contradicted_findings(self, candidate: Any) -> List[str]:
        values: List[str] = []
        for key in (
            "hard_contradicted_evidence",
            "soft_contradicted_evidence",
            "contradicted_evidence",
            "eligibility_blockers",
        ):
            values.extend(self._texts(getattr(candidate, key, []) or []))
        return list(dict.fromkeys(values))

    def _matched_diagnostic_patterns(self, candidate: Any) -> List[str]:
        result: List[str] = []
        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(pattern, dict) or not bool(pattern.get("matched", False)):
                continue
            effect = dict(pattern.get("effect") or {})
            if str(effect.get("eligibility") or pattern.get("eligibility") or "") == PRIMARY_ELIGIBLE:
                pattern_id = str(pattern.get("pattern_id") or "")
                if pattern_id:
                    result.append(pattern_id)
        return list(dict.fromkeys(result))

    def _matched_bridge_patterns(self, candidate: Any) -> List[str]:
        result: List[str] = []
        for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
            if isinstance(assertion, dict):
                value = str(assertion.get("canonical_pattern") or "")
                if value:
                    result.append(value)
        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(pattern, dict):
                continue
            pattern_type = str(pattern.get("pattern_type") or pattern.get("role") or "")
            if "bridge" in pattern_type and bool(pattern.get("matched", False)):
                value = str(pattern.get("pattern_id") or "")
                if value:
                    result.append(value)
        return list(dict.fromkeys(result))

    def _actionable_gaps(self, candidate: Any) -> List[str]:
        values = list(self._texts(getattr(candidate, "required_gaps", []) or []))
        for gap in getattr(candidate, "evidence_gaps", []) or []:
            if isinstance(gap, dict):
                gap_id = str(gap.get("gap_id") or gap.get("target_evidence") or "")
                if gap_id:
                    values.append(gap_id)
        return list(dict.fromkeys(values))

    @staticmethod
    def _texts(values: Iterable[Any]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values or []:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result
