"""Primary diagnosis eligibility and evidence-contribution scoring."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


PRIMARY_ELIGIBLE = "PrimaryEligible"
DEFERRED = "Deferred"
DIFFERENTIAL_ONLY = "DifferentialOnly"
EXCLUDED = "Excluded"

NEEDS_ANCHOR = "NeedsAnchor"
CONFLICT_NEEDS_ADJUDICATION = "ConflictNeedsAdjudication"
HARD_CONTRADICTION = "HardContradiction"
NO_SUPPORTING_EVIDENCE = "NoSupportingEvidence"
WEAK_DIFFERENTIAL_SIGNAL = "WeakDifferentialSignal"
INSUFFICIENT_EXPLANATION = "InsufficientExplanation"
ANCHORS_SATISFIED = "AnchorsSatisfied"
PATTERN_CONTRADICTED = "PatternContradicted"


_ACUTE_PROSTATITIS_ANCHORS = {
    "diagnosis:急性细菌性前列腺炎",
    "prostate_tenderness",
    "perineal_pain",
    "pelvic_pain",
    "dysuria",
    "urinary_frequency",
    "urinary_urgency",
    "bacteriuria",
    "urine_culture_positive",
    "leukocyte_esterase_positive",
    "nitrite_positive",
}

_ACUTE_PROSTATITIS_LOCALIZING_ANCHORS = {
    "prostate_tenderness",
    "perineal_pain",
    "pelvic_pain",
    "dysuria",
    "urinary_frequency",
    "urinary_urgency",
}

_ACUTE_PROSTATITIS_BACTERIAL_ANCHORS = {
    "bacteriuria",
    "urine_culture_positive",
    "leukocyte_esterase_positive",
    "nitrite_positive",
}

_ACUTE_PROSTATITIS_INFLAMMATION_SUPPORT = {
    "pyuria",
}

_ACUTE_PROSTATITIS_BACTERIAL_BLOCKERS = {
    "urine_culture_no_growth",
    "leukocyte_esterase_negative",
    "nitrite_negative",
    "urine_wbc_normal",
}

_PULMONARY_CRYPTOCOCCOSIS_ANCHORS = {
    "diagnosis:肺隐球菌病",
    "cryptococcal_antigen_positive",
    "crag_positive",
    "fungal_pneumonia",
    "fungal_culture_positive",
    "immunocompromised",
    "pulmonary_nodule",
    "cavitary_lesion",
}

_MYCOPLASMA_PNEUMONIA_ANCHORS = {
    "diagnosis:支原体肺炎",
    "mycoplasma_naat_positive",
    "mycoplasma_antibody_positive",
    "interstitial_infiltrate",
}


@dataclass
class EligibilityResult:
    diagnosis: str
    status: str
    reason: str
    missing_required_anchors: List[str] = field(default_factory=list)
    satisfied_required_anchors: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evidence_contributions: List[Dict[str, Any]] = field(default_factory=list)
    evidence_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    positive_evidence_score: float = 0.0
    evidence_specificity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceSpecificityCalculator:
    """Approximate evidence specificity from global knowledge plus the local pool."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge
        self.total_diseases = 0
        self.global_support: Dict[str, Set[str]] = {}
        self.match_strengths: Dict[str, Dict[str, float]] = {}
        self._build_global_index()

    def apply(self, candidates: Sequence[Any], evidence: Any = None) -> None:
        observations = list(getattr(evidence, "observations", []) or [])
        obs_by_finding = self._best_observation_by_finding(observations)
        local_counts = self._local_support_counts(candidates)
        pool_size = max(1, len([item for item in candidates if item]))
        for candidate in candidates or []:
            if not candidate:
                continue
            contributions = self._candidate_contributions(
                candidate,
                obs_by_finding,
                local_counts,
                pool_size,
            )
            positive = round(min(1.0, sum(item["contribution"] for item in contributions)), 4)
            specificity = round(
                max((item["combined_es"] for item in contributions), default=0.0),
                4,
            )
            setattr(candidate, "evidence_contributions", contributions)
            setattr(candidate, "positive_evidence_score", positive)
            setattr(candidate, "evidence_specificity_score", specificity)
            components = getattr(candidate, "component_scores", None)
            if isinstance(components, dict):
                components["positive_evidence_score"] = positive
                components["evidence_specificity_score"] = specificity

    def _candidate_contributions(
        self,
        candidate: Any,
        obs_by_finding: Dict[str, Any],
        local_counts: Dict[str, int],
        pool_size: int,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        matched = list(dict.fromkeys(getattr(candidate, "matched_evidence", []) or []))
        diagnostic = set(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        core = set(getattr(candidate, "core_matched_evidence", []) or [])
        generic = set(getattr(candidate, "generic_matched_evidence", []) or [])
        temporal = self._temporal_fit(candidate)
        for finding in matched:
            text = str(finding or "")
            if not text:
                continue
            observation = obs_by_finding.get(text)
            confidence = self._confidence(observation)
            reliability = self._source_reliability(observation)
            match = self._match_strength(diagnosis, text, diagnostic, core, generic)
            global_es = self._global_specificity(text, observation)
            local_es = self._local_specificity(text, local_counts, pool_size)
            combined = max(0.0, min(1.0, 0.7 * global_es + 0.3 * local_es))
            contribution = combined * match * confidence * reliability * temporal
            result.append(
                {
                    "finding": text,
                    "global_es": round(global_es, 4),
                    "local_es": round(local_es, 4),
                    "combined_es": round(combined, 4),
                    "match_strength": round(match, 4),
                    "confidence": round(confidence, 4),
                    "source_reliability": round(reliability, 4),
                    "temporal_fit": round(temporal, 4),
                    "contribution": round(contribution, 4),
                    "source": str(getattr(observation, "source", "") or ""),
                    "evidence_level": str(getattr(observation, "evidence_level", "") or ""),
                }
            )
        result.sort(
            key=lambda item: (
                item["contribution"],
                item["combined_es"],
                item["match_strength"],
            ),
            reverse=True,
        )
        return result[:12]

    def _build_global_index(self) -> None:
        entries = getattr(self.knowledge, "entries", {}) or {}
        self.total_diseases = max(1, len(entries))
        for name, entry in entries.items():
            diagnosis = str(name or entry.get("name") or "")
            for spec in list(entry.get("supporting_evidence", []) or []):
                for finding in self._spec_findings(spec):
                    self.global_support.setdefault(finding, set()).add(diagnosis)
                    self._record_match_strength(diagnosis, finding, spec)
            for group in entry.get("required_groups", []) or []:
                for spec in group or []:
                    for finding in self._spec_findings(spec):
                        self.global_support.setdefault(finding, set()).add(diagnosis)
                        self._record_match_strength(diagnosis, finding, spec, floor=0.75)

    def _record_match_strength(
        self,
        diagnosis: str,
        finding: str,
        spec: Any,
        *,
        floor: float = 0.15,
    ) -> None:
        if not diagnosis or not finding:
            return
        weight = floor
        if isinstance(spec, dict):
            try:
                weight = float(spec.get("weight", weight) or weight)
            except (TypeError, ValueError):
                weight = floor
        strength = max(floor, min(1.0, weight * 2.5))
        by_finding = self.match_strengths.setdefault(diagnosis, {})
        by_finding[finding] = max(by_finding.get(finding, 0.0), strength)

    @staticmethod
    def _spec_findings(spec: Any) -> List[str]:
        if isinstance(spec, str):
            return [spec] if spec else []
        if not isinstance(spec, dict):
            return []
        values = []
        for key in ("finding", "concept", "evidence"):
            value = str(spec.get(key) or "").strip()
            if value:
                values.append(value)
        return values

    @staticmethod
    def _best_observation_by_finding(observations: Sequence[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in observations or []:
            finding = str(getattr(item, "finding", "") or "")
            if not finding:
                continue
            current = result.get(finding)
            if current is None:
                result[finding] = item
                continue
            if EvidenceSpecificityCalculator._observation_quality(item) > EvidenceSpecificityCalculator._observation_quality(current):
                result[finding] = item
        return result

    @staticmethod
    def _observation_quality(item: Any) -> float:
        try:
            confidence = float(getattr(item, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            info = float(getattr(item, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            info = 0.0
        return confidence * 0.4 + info * 0.6

    @staticmethod
    def _local_support_counts(candidates: Sequence[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for candidate in candidates or []:
            for finding in set(getattr(candidate, "matched_evidence", []) or []):
                text = str(finding or "")
                if text:
                    counts[text] = counts.get(text, 0) + 1
        return counts

    def _global_specificity(self, finding: str, observation: Any) -> float:
        try:
            info = float(getattr(observation, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            info = 0.0
        supporters = self.global_support.get(finding) or set()
        if not supporters:
            return max(0.05, min(1.0, info or 0.08))
        rarity = 1.0 - (
            math.log(1.0 + len(supporters)) / max(1.0, math.log(1.0 + self.total_diseases))
        )
        return max(0.05, min(1.0, 0.65 * rarity + 0.35 * (info or 0.08)))

    @staticmethod
    def _local_specificity(
        finding: str,
        local_counts: Dict[str, int],
        pool_size: int,
    ) -> float:
        count = max(1, int(local_counts.get(finding, 1) or 1))
        if pool_size <= 1:
            return 0.5
        return max(0.0, min(1.0, 1.0 - ((count - 1) / max(1, pool_size - 1))))

    def _match_strength(
        self,
        diagnosis: str,
        finding: str,
        diagnostic: Set[str],
        core: Set[str],
        generic: Set[str],
    ) -> float:
        if finding in diagnostic:
            return 1.0
        if finding in core:
            return max(0.75, self.match_strengths.get(diagnosis, {}).get(finding, 0.0))
        if finding in generic:
            return max(0.15, min(0.35, self.match_strengths.get(diagnosis, {}).get(finding, 0.25)))
        return self.match_strengths.get(diagnosis, {}).get(finding, 0.45)

    @staticmethod
    def _confidence(observation: Any) -> float:
        try:
            return max(0.0, min(1.0, float(getattr(observation, "confidence", 0.75) or 0.75)))
        except (TypeError, ValueError):
            return 0.75

    @staticmethod
    def _source_reliability(observation: Any) -> float:
        source = str(getattr(observation, "source", "") or "").lower()
        if source == "reasoning_inference":
            return 0.22
        if source == "raw_case_finding":
            return 0.55
        if any(token in source for token in ("病理", "病原", "培养", "naat", "afb", "pcr")):
            return 0.98
        if any(token in source for token in ("ct", "mri", "x线", "影像", "超声", "心电图", "ecg")):
            return 0.9
        if any(token in source for token in ("血", "尿", "实验", "检测", "镁负荷", "维生素", "pth")):
            return 0.86
        if any(token in source for token in ("体格", "查体", "医生观察")):
            return 0.74
        if any(token in source for token in ("问诊", "主诉", "patient")):
            return 0.6
        return 0.68

    @staticmethod
    def _temporal_fit(candidate: Any) -> float:
        components = getattr(candidate, "component_scores", {}) or {}
        try:
            temporal = float(components.get("temporal", 0.0) or 0.0)
        except (TypeError, ValueError):
            temporal = 0.0
        if temporal <= 0.0:
            return 1.0
        return max(0.7, min(1.0, 0.7 + 0.3 * temporal))


class DiagnosisEligibilityGate:
    """The single authority for whether a candidate may become final primary."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge
        self.specificity_calculator = EvidenceSpecificityCalculator(knowledge)

    def evaluate_all(
        self,
        candidates: Sequence[Any],
        evidence: Any = None,
    ) -> Dict[str, Any]:
        self.specificity_calculator.apply(candidates, evidence)
        results = [self.evaluate(candidate) for candidate in candidates or [] if candidate]
        for candidate, result in zip([item for item in candidates or [] if item], results):
            self.apply_result(candidate, result)
        return self.summary(results)

    def evaluate(self, candidate: Any) -> EligibilityResult:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        missing = list(dict.fromkeys(getattr(candidate, "required_gaps", []) or []))
        satisfied = self._satisfied_anchors(candidate)
        blockers: List[str] = []
        if getattr(candidate, "hard_contradiction", False):
            blockers.extend(getattr(candidate, "hard_contradicted_evidence", []) or [])
            return self._result(candidate, EXCLUDED, HARD_CONTRADICTION, missing, satisfied, blockers)
        if not getattr(candidate, "matched_evidence", None):
            return self._result(candidate, DIFFERENTIAL_ONLY, NO_SUPPORTING_EVIDENCE, missing, satisfied, blockers)
        if getattr(candidate, "unresolved_evidence_conflict", False):
            blockers.append("unresolved_reasoning_structured_evidence_conflict")
            return self._result(candidate, DEFERRED, CONFLICT_NEEDS_ADJUDICATION, missing, satisfied, blockers)
        blocking_pattern = self._blocking_evidence_pattern(candidate)
        if blocking_pattern:
            blockers.extend(blocking_pattern.get("blockers", []))
            self._append_evidence_pattern(candidate, blocking_pattern)
            return self._result(candidate, DIFFERENTIAL_ONLY, PATTERN_CONTRADICTED, missing, satisfied, blockers)
        if bool(getattr(candidate, "differential_only", False)):
            reason = str(getattr(candidate, "differential_only_reason", "") or WEAK_DIFFERENTIAL_SIGNAL)
            return self._result(candidate, DIFFERENTIAL_ONLY, reason, missing, satisfied, blockers)
        if missing:
            if self._deferred_worth_followup(candidate):
                return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
            return self._result(candidate, DIFFERENTIAL_ONLY, WEAK_DIFFERENTIAL_SIGNAL, missing, satisfied, blockers)
        sanity_gap = self._diagnosis_anchor_sanity_gap(candidate)
        if sanity_gap:
            missing = list(dict.fromkeys(list(missing) + [sanity_gap]))
            return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
        if self._insufficient_explanation(candidate):
            return self._result(candidate, DIFFERENTIAL_ONLY, INSUFFICIENT_EXPLANATION, missing, satisfied, blockers)
        return self._result(candidate, PRIMARY_ELIGIBLE, ANCHORS_SATISFIED, missing, satisfied, blockers)

    def apply_result(self, candidate: Any, result: EligibilityResult) -> None:
        setattr(candidate, "eligibility_status", result.status)
        setattr(candidate, "eligibility_reason", result.reason)
        setattr(candidate, "missing_required_anchors", list(result.missing_required_anchors))
        setattr(candidate, "satisfied_required_anchors", list(result.satisfied_required_anchors))
        setattr(candidate, "eligibility_blockers", list(result.blockers))
        setattr(candidate, "evidence_pattern_matches", list(result.evidence_pattern_matches))
        setattr(candidate, "positive_evidence_score", result.positive_evidence_score)
        setattr(candidate, "evidence_specificity_score", result.evidence_specificity_score)

    @staticmethod
    def summary(results: Sequence[EligibilityResult]) -> Dict[str, Any]:
        distribution: Dict[str, int] = {}
        deferred: List[str] = []
        excluded: List[str] = []
        primary: List[str] = []
        differential: List[str] = []
        for result in results:
            distribution[result.status] = distribution.get(result.status, 0) + 1
            if result.status == PRIMARY_ELIGIBLE:
                primary.append(result.diagnosis)
            elif result.status == DEFERRED:
                deferred.append(result.diagnosis)
            elif result.status == EXCLUDED:
                excluded.append(result.diagnosis)
            elif result.status == DIFFERENTIAL_ONLY:
                differential.append(result.diagnosis)
        return {
            "eligibility_distribution": distribution,
            "primary_eligible_candidates": primary,
            "deferred_anchor_candidates": deferred,
            "excluded_candidates": excluded,
            "differential_only_candidates": differential,
        }

    def _result(
        self,
        candidate: Any,
        status: str,
        reason: str,
        missing: Sequence[str],
        satisfied: Sequence[str],
        blockers: Sequence[str],
    ) -> EligibilityResult:
        return EligibilityResult(
            diagnosis=str(getattr(candidate, "diagnosis", "") or ""),
            status=status,
            reason=reason,
            missing_required_anchors=list(missing),
            satisfied_required_anchors=list(satisfied),
            blockers=list(blockers),
            evidence_contributions=list(getattr(candidate, "evidence_contributions", []) or []),
            evidence_pattern_matches=list(getattr(candidate, "evidence_pattern_matches", []) or []),
            positive_evidence_score=float(getattr(candidate, "positive_evidence_score", 0.0) or 0.0),
            evidence_specificity_score=float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0),
        )

    @staticmethod
    def _satisfied_anchors(candidate: Any) -> List[str]:
        values = []
        for finding in list(getattr(candidate, "diagnostic_matched_evidence", []) or []):
            values.append(str(finding))
        for finding in list(getattr(candidate, "core_matched_evidence", []) or []):
            values.append(str(finding))
        return list(dict.fromkeys(item for item in values if item))

    def _deferred_worth_followup(self, candidate: Any) -> bool:
        entry = self._entry(candidate)
        if entry.get("discriminating_exams") or entry.get("strong_verification_exams") or entry.get("required_exams"):
            return True
        if float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45:
            return True
        if float(getattr(candidate, "core_explanatory_coverage", 0.0) or 0.0) >= 0.35:
            return True
        if float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.45:
            return True
        if float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0) >= 0.65:
            return True
        return bool(getattr(candidate, "core_matched_evidence", None) or getattr(candidate, "diagnostic_matched_evidence", None))

    def _diagnosis_anchor_sanity_gap(self, candidate: Any) -> str:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        entry = self._entry(candidate)
        category = str(entry.get("category") or getattr(candidate, "category", "") or "")
        if diagnosis == "肺隐球菌病":
            matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
            if matched & _PULMONARY_CRYPTOCOCCOSIS_ANCHORS:
                return ""
            return "pulmonary_cryptococcosis_requires_fungal_or_cryptococcal_anchor"
        if diagnosis == "支原体肺炎":
            matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
            if matched & _MYCOPLASMA_PNEUMONIA_ANCHORS:
                return ""
            return "mycoplasma_pneumonia_requires_pathogen_or_interstitial_anchor"
        if diagnosis != "急性细菌性前列腺炎" and category != "acute_bacterial_prostate":
            return ""
        matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
        if matched & _ACUTE_PROSTATITIS_ANCHORS:
            return ""
        return "acute_bacterial_prostatitis_requires_urinary_or_prostate_anchor"

    def _blocking_evidence_pattern(self, candidate: Any) -> Dict[str, Any]:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        entry = self._entry(candidate)
        category = str(entry.get("category") or getattr(candidate, "category", "") or "")
        if diagnosis != "急性细菌性前列腺炎" and category != "acute_bacterial_prostate":
            return {}
        matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
        contradicted = {
            str(item)
            for item in (
                list(getattr(candidate, "soft_contradicted_evidence", []) or [])
                + list(getattr(candidate, "hard_contradicted_evidence", []) or [])
                + list(getattr(candidate, "contradicted_evidence", []) or [])
            )
        }
        bacterial_positive = sorted(matched & _ACUTE_PROSTATITIS_BACTERIAL_ANCHORS)
        if bacterial_positive:
            return {}
        bacterial_blockers = sorted((matched | contradicted) & _ACUTE_PROSTATITIS_BACTERIAL_BLOCKERS)
        if not bacterial_blockers:
            return {}
        localizing = sorted(matched & _ACUTE_PROSTATITIS_LOCALIZING_ANCHORS)
        inflammation_only = sorted(matched & _ACUTE_PROSTATITIS_INFLAMMATION_SUPPORT)
        multiple_negative_markers = len(bacterial_blockers) >= 2
        lacks_localizing_anchor = not bool(localizing)
        if not inflammation_only and not (multiple_negative_markers and lacks_localizing_anchor):
            return {}
        if not multiple_negative_markers and not lacks_localizing_anchor:
            return {}
        return {
            "pattern": "acute_bacterial_prostatitis_negative_urine_pattern",
            "role": "negative_pattern",
            "matched": sorted(set(inflammation_only + localizing)),
            "blockers": bacterial_blockers,
            "action": "downgrade_to_differential_only",
            "reason": (
                "pyuria is urinary inflammation support, not a bacterial "
                "prostatitis anchor when bacterial urine markers are negative"
            ),
        }

    @staticmethod
    def _append_evidence_pattern(candidate: Any, pattern: Dict[str, Any]) -> None:
        existing = list(getattr(candidate, "evidence_pattern_matches", []) or [])
        existing.append(dict(pattern))
        setattr(candidate, "evidence_pattern_matches", existing)

    @staticmethod
    def _insufficient_explanation(candidate: Any) -> bool:
        if not bool(getattr(candidate, "required_met", False)):
            return False
        if float(getattr(candidate, "diagnostic_evidence_score", 0.0) or 0.0) >= 0.25:
            return False
        if float(getattr(candidate, "core_evidence_score", 0.0) or 0.0) >= 0.35:
            return False
        if int(getattr(candidate, "residual_core_evidence_count", 0) or 0) >= 3:
            return float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.45
        return False

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if not self.knowledge:
            return {}
        try:
            return dict(self.knowledge.get(str(getattr(candidate, "diagnosis", "") or "")) or {})
        except Exception:
            return {}


def eligibility_status(candidate: Any) -> str:
    return str(getattr(candidate, "eligibility_status", "") or "")


def is_primary_eligible(candidate: Any) -> bool:
    return eligibility_status(candidate) == PRIMARY_ELIGIBLE


def is_deferred(candidate: Any) -> bool:
    return eligibility_status(candidate) == DEFERRED


def is_excluded(candidate: Any) -> bool:
    return eligibility_status(candidate) == EXCLUDED
