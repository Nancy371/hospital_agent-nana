"""Evidence-first candidate scoring and final diagnosis adjudication."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .candidate_generator import CandidateGenerator, CandidatePool
from .clinical_evidence import EvidenceBundle, Observation
from .diagnosis_resolver import DiagnosisResolution, OpenWorldDiagnosisResolver


_SECONDARY_MANIFESTATION_DIAGNOSES = {
    "心律失常",
    "心力衰竭",
    "肺动脉高压",
}


@dataclass
class CandidateScore:
    diagnosis: str
    score: float
    support_score: float
    source_prior: float
    explanation_score: float
    coverage_score: float
    residual_score: float
    contradiction_penalty: float
    required_met: bool
    hard_contradiction: bool
    matched_evidence: List[str] = field(default_factory=list)
    contradicted_evidence: List[str] = field(default_factory=list)
    required_gaps: List[str] = field(default_factory=list)
    residual_evidence: List[str] = field(default_factory=list)
    component_scores: Dict[str, float] = field(default_factory=dict)
    candidate_sources: List[Dict[str, Any]] = field(default_factory=list)
    diagnosis_type: str = "disease"
    parent_diagnosis: str = ""
    specificity: float = 0.5
    causal_relation_to_selected: str = ""
    differential_only: bool = False
    differential_only_reason: str = ""

    @property
    def trusted(self) -> bool:
        return (
            self.required_met
            and not self.hard_contradiction
            and bool(self.matched_evidence)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisDecision:
    final_diagnoses: List[str]
    trusted_diagnoses: List[str]
    candidates: List[CandidateScore]
    unexplained_evidence: List[str]
    confidence: float
    margin: float
    low_confidence: bool
    evidence_reasoning: str = ""
    name_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_candidates: List[str] = field(default_factory=list)
    differential_only_diagnoses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_diagnoses": list(self.final_diagnoses),
            "trusted_diagnoses": list(self.trusted_diagnoses),
            "candidates": [item.to_dict() for item in self.candidates],
            "unexplained_evidence": list(self.unexplained_evidence),
            "confidence": self.confidence,
            "margin": self.margin,
            "low_confidence": self.low_confidence,
            "evidence_reasoning": self.evidence_reasoning,
            "name_resolutions": list(self.name_resolutions),
            "unresolved_candidates": list(self.unresolved_candidates),
            "differential_only_diagnoses": list(self.differential_only_diagnoses),
        }


class DiagnosticKnowledgeBase:
    """Load official names, controlled extensions, profiles, and evidence rules."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.catalog_path = os.path.join(ref_dir, "diseases_catalog.json")
        self.extensions_path = os.path.join(ref_dir, "submission_diagnosis_extensions.json")
        self.knowledge_path = os.path.join(ref_dir, "diagnostic_knowledge.json")
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, str] = {}
        self.official_names: Set[str] = set()
        self.extension_names: Set[str] = set()
        self.knowledge_version = ""
        self.source_registry: Dict[str, Any] = {}
        self._load()

    @property
    def allowed_names(self) -> List[str]:
        return sorted(self.official_names | self.extension_names)

    def normalize_name(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        if text in self.entries and text in (self.official_names | self.extension_names):
            return text
        if text in self.aliases:
            return self.aliases[text]
        lowered = text.lower()
        for alias, name in self.aliases.items():
            if alias.lower() == lowered:
                return name
        # Avoid broad substring normalization except for explicit suffixes from LLM output.
        stripped = text.replace("诊断", "").replace("可能", "").strip(" ：:，,。")
        return self.aliases.get(stripped)

    def is_allowed(self, name: Any) -> bool:
        normalized = self.normalize_name(name)
        return bool(normalized and normalized in (self.official_names | self.extension_names))

    def get(self, name: Any) -> Dict[str, Any]:
        normalized = self.normalize_name(name) or str(name or "")
        return self.entries.get(normalized, {})

    def get_treatment_protocols(self, diagnoses: Iterable[Any]) -> List[str]:
        protocols: List[str] = []
        for diagnosis in diagnoses or []:
            for item in self.get(diagnosis).get("treatment_protocol", []) or []:
                text = str(item).strip()
                if text and text not in protocols:
                    protocols.append(text)
        return protocols

    def get_contraindications(self, diagnoses: Iterable[Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for diagnosis in diagnoses or []:
            for item in self.get(diagnosis).get("contraindications", []) or []:
                if isinstance(item, str):
                    item = {"term": item}
                if isinstance(item, dict) and item not in result:
                    result.append(dict(item))
        return result

    def _load(self) -> None:
        catalog = _read_json(self.catalog_path, {}).get("diseases", [])
        for item in catalog:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            self.official_names.add(name)
            self.aliases[name] = name
            self.entries[name] = self._base_entry(name)

        extensions = _read_json(self.extensions_path, {}).get("extensions", [])
        extension_meta: Dict[str, Dict[str, Any]] = {}
        for item in extensions:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            self.extension_names.add(name)
            extension_meta[name] = dict(item)
            self.aliases[name] = name
            for alias in item.get("aliases", []) or []:
                if str(alias).strip():
                    self.aliases[str(alias).strip()] = name
            entry = self._base_entry(name)
            entry["parent_diagnosis"] = str(item.get("parent_catalog_name") or "")
            entry["specificity"] = float(item.get("specificity", 0.8) or 0.8)
            entry["sources"] = list(item.get("sources", []) or [])
            self.entries[name] = entry

        profiles: Dict[str, Dict[str, Any]] = {}
        if os.path.isdir(self.ref_dir):
            for filename in sorted(os.listdir(self.ref_dir)):
                if not filename.startswith("disease_profiles") or not filename.endswith(".json"):
                    continue
                for profile in _read_json(os.path.join(self.ref_dir, filename), {}).get("profiles", []):
                    name = str(profile.get("name") or "").strip()
                    if name:
                        profiles[name] = dict(profile)

        for name, entry in list(self.entries.items()):
            profile = profiles.get(name, {})
            if profile.get("department"):
                entry["department"] = str(profile.get("department") or "")
            entry["aliases"] = list(dict.fromkeys(profile.get("aliases", []) or []))
            for alias in entry["aliases"]:
                if str(alias).strip():
                    self.aliases[str(alias).strip()] = name
            entry["discriminating_exams"] = list(
                dict.fromkeys(
                    list(profile.get("strong_verification_exams", []) or [])
                    + list(profile.get("required_exams", []) or [])
                )
            )
            entry["treatment_protocol"] = list(profile.get("treatment_principles", []) or [])
            entry["avoid_mistakes"] = list(profile.get("avoid_mistakes", []) or [])
            entry["supporting_evidence"] = self._profile_support(profile, name)

        for knowledge_payload in self._iter_knowledge_payloads():
            if knowledge_payload.get("knowledge_version"):
                self.knowledge_version = str(knowledge_payload.get("knowledge_version") or "")
            self.source_registry.update(dict(knowledge_payload.get("source_registry") or {}))
            overrides = knowledge_payload.get("diseases", [])
            for override in overrides:
                name = str(override.get("name") or "").strip()
                if name not in self.entries:
                    continue
                merged = dict(self.entries[name])
                for key, value in override.items():
                    if key == "name":
                        continue
                    if key == "supporting_evidence":
                        merged[key] = _dedupe_specs(
                            list(merged.get(key, [])) + list(value or [])
                        )
                    elif key in {
                        "treatment_protocol",
                        "contraindications",
                        "sources",
                        "contradictions",
                        "causes",
                        "caused_by",
                    }:
                        merged[key] = _dedupe_objects(
                            list(merged.get(key, [])) + list(value or [])
                        )
                    else:
                        merged[key] = value
                self.entries[name] = merged

        # A direct positive or negative mention is a generic evidence source for every disease.
        for name, entry in self.entries.items():
            entry["supporting_evidence"] = _dedupe_specs(
                [{"finding": f"diagnosis:{name}", "weight": 0.65}]
                + list(entry.get("supporting_evidence", []))
            )

    def _iter_knowledge_payloads(self) -> Iterable[Dict[str, Any]]:
        paths = [self.knowledge_path]
        if os.path.isdir(self.ref_dir):
            for filename in sorted(os.listdir(self.ref_dir)):
                if (
                    filename.startswith("diagnostic_knowledge_")
                    and filename.endswith(".json")
                ):
                    paths.append(os.path.join(self.ref_dir, filename))
        for path in paths:
            payload = _read_json(path, {})
            if isinstance(payload, dict):
                yield payload

    @staticmethod
    def _base_entry(name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "diagnosis_type": "disease",
            "parent_diagnosis": "",
            "supporting_evidence": [],
            "required_groups": [],
            "contradictions": [],
            "discriminating_exams": [],
            "specificity": 0.5,
            "treatment_protocol": [],
            "contraindications": [],
            "suppress_diagnoses": [],
            "causes": [],
            "caused_by": [],
            "sources": [],
            "source_version": "",
            "department": "",
        }

    @staticmethod
    def _profile_support(profile: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for symptom in profile.get("common_symptoms", []) or []:
            text = str(symptom).strip()
            if text:
                specs.append({"terms": [text], "weight": 0.2})
        for red_flag in profile.get("red_flags", []) or []:
            text = str(red_flag).strip()
            if text:
                specs.append({"terms": [text], "weight": 0.24})
        return specs


class DiagnosisDecisionEngine:
    """Score every allowed diagnosis and arbitrate a small final diagnosis set."""

    def __init__(self, config: Dict[str, Any], ref_dir: str = "data/ref_data"):
        section = config.get("diagnosis", {}) or {}
        self.trusted_threshold = float(section.get("trusted_threshold", 0.65) or 0.65)
        self.differential_threshold = float(section.get("differential_threshold", 0.35) or 0.35)
        self.margin_threshold = float(section.get("margin_threshold", 0.12) or 0.12)
        self.max_final_diagnoses = int(section.get("max_final_diagnoses", 3) or 3)
        self.candidate_limit = int(section.get("candidate_limit", 12) or 12)
        self.etiology_priority_bonus = float(
            section.get("etiology_priority_bonus", 0.08) or 0.08
        )
        self.etiology_close_margin = float(
            section.get("etiology_close_margin", 0.12) or 0.12
        )
        self.max_evidence_gap_targets = int(
            section.get("max_evidence_gap_targets", 2) or 2
        )
        self.residual_drop_threshold = float(
            section.get("residual_drop_threshold", 0.58) or 0.58
        )
        self.evidence_gap_coverage_threshold = float(
            section.get("evidence_gap_coverage_threshold", 0.32) or 0.32
        )
        self.evidence_gap_residual_threshold = float(
            section.get("evidence_gap_residual_threshold", 0.72) or 0.72
        )
        self.required_group_policy = str(section.get("required_group_policy") or "gap_only")
        self.weights = self._load_weights(section.get("weights") or {})
        self.knowledge = DiagnosticKnowledgeBase(ref_dir=ref_dir)
        self.resolver = OpenWorldDiagnosisResolver(self.knowledge, config=config)
        self.candidate_generator = CandidateGenerator(self.knowledge, self.resolver)

    @staticmethod
    def _load_weights(configured: Dict[str, Any]) -> Dict[str, float]:
        defaults = {
            "evidence": 0.52,
            "prior": 0.10,
            "specificity": 0.08,
            "explain": 0.16,
            "exam_match": 0.06,
            "temporal": 0.03,
            "age": 0.02,
            "risk": 0.03,
            "residual": 0.10,
            "contradiction": 1.0,
        }
        for key, default in list(defaults.items()):
            try:
                defaults[key] = float(configured.get(key, default))
            except (AttributeError, TypeError, ValueError):
                defaults[key] = default
        return defaults

    def decide(
        self,
        llm_result: Optional[Dict[str, Any]],
        rag_chunks: Optional[Sequence[Dict[str, Any]]],
        evidence: EvidenceBundle,
    ) -> DiagnosisDecision:
        candidate_pool = self.candidate_generator.generate(
            evidence_graph=evidence.to_graph(),
            llm_result=llm_result or {},
            rag_chunks=rag_chunks or [],
            evidence=evidence,
        )
        return self.rank(candidate_pool, evidence)

    def rank(
        self,
        candidate_pool: CandidatePool,
        evidence: EvidenceBundle,
    ) -> DiagnosisDecision:
        priors = candidate_pool.priors()
        sources_by_name = candidate_pool.sources_by_name()
        scores = [
            self._score_entry(
                entry,
                priors.get(name, 0.0),
                evidence,
                candidate_sources=sources_by_name.get(name, []),
            )
            for name, entry in self.knowledge.entries.items()
        ]
        scores = self._sort_candidates(scores)
        self._clear_submission_marks(scores)

        trusted_pool = [
            item for item in scores
            if item.trusted and item.score >= self.trusted_threshold
        ]
        selected = self._select_final(trusted_pool)
        selected = self._append_independent_states(selected, scores)
        trusted_names = [item.diagnosis for item in selected]

        if not selected:
            supported = [
                item
                for item in scores
                if item.trusted and item.score >= self.differential_threshold
            ]
            if supported:
                selected = [supported[0]]
                selected = self._append_independent_states(selected, scores)
            else:
                fallback = self._fallback_candidate(priors, scores)
                selected = [fallback] if fallback else []

        self._annotate_causal_relations(scores, selected)
        differential_only = self.differential_only_details(scores)
        final_names = [item.diagnosis for item in selected][: self.max_final_diagnoses]
        top_score = scores[0].score if scores else 0.0
        margin = top_score - scores[1].score if len(scores) > 1 else top_score
        explained = set()
        for item in selected:
            explained.update(item.matched_evidence)
        unexplained = [
            item.finding for item in evidence.major()
            if item.finding not in explained
        ]
        unexplained = list(dict.fromkeys(unexplained))[:12]
        low_confidence = (
            not trusted_names
            or top_score < self.trusted_threshold
            or margin < self.margin_threshold
            or bool(unexplained)
            or any(item.hard_contradiction for item in selected)
        )
        reasoning = self._reasoning(selected, unexplained)
        return DiagnosisDecision(
            final_diagnoses=final_names,
            trusted_diagnoses=trusted_names,
            # Keep all scored diagnoses in the local audit object. Prompt and
            # log renderers apply their own display limits, while Critic and
            # replay can still inspect a low-ranked hard contradiction.
            candidates=scores,
            unexplained_evidence=unexplained,
            confidence=round(top_score, 4),
            margin=round(margin, 4),
            low_confidence=low_confidence,
            evidence_reasoning=reasoning,
            name_resolutions=list(candidate_pool.name_resolutions),
            unresolved_candidates=list(candidate_pool.unresolved_candidates),
            differential_only_diagnoses=differential_only,
        )

    def filter_final_diagnoses(
        self,
        diagnosis_names: Sequence[str],
        scores: Sequence[CandidateScore],
    ) -> List[CandidateScore]:
        self._clear_submission_marks(scores)
        score_by_name = {item.diagnosis: item for item in scores}
        pool = [
            score_by_name[name]
            for name in dict.fromkeys(str(item).strip() for item in diagnosis_names if str(item).strip())
            if name in score_by_name
            and score_by_name[name].trusted
            and not score_by_name[name].hard_contradiction
        ]
        selected = self._select_final(self._sort_candidates(pool))
        selected = self._append_independent_states(selected, scores)
        self._annotate_causal_relations(scores, selected)
        return selected[: self.max_final_diagnoses]

    def apply_to_result(
        self,
        result: Optional[Dict[str, Any]],
        decision: DiagnosisDecision,
        evidence: EvidenceBundle,
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        decision.differential_only_diagnoses = self.differential_only_details(decision.candidates)
        fixed["diagnosis"] = list(decision.final_diagnoses)
        fixed["_trusted_diagnoses"] = list(decision.trusted_diagnoses)
        fixed["_diagnosis_decision"] = decision.to_dict()
        fixed["_diagnosis_name_resolution"] = list(decision.name_resolutions)
        fixed["_unresolved_diagnosis_candidates"] = list(decision.unresolved_candidates)
        fixed["_evidence_items"] = [item.to_dict() for item in evidence.observations]
        reasoning = str(fixed.get("reasoning") or "").strip()
        if decision.evidence_reasoning and decision.evidence_reasoning not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + decision.evidence_reasoning
        fixed["reasoning"] = reasoning
        return fixed

    def _sort_candidates(
        self, candidates: Sequence[CandidateScore]
    ) -> List[CandidateScore]:
        return sorted(candidates, key=self._candidate_sort_key, reverse=True)

    def _candidate_sort_key(self, candidate: CandidateScore) -> Tuple[float, int, float, float, float]:
        return (
            self._adjudication_score(candidate),
            self._diagnosis_type_rank(candidate),
            1.0 - candidate.residual_score,
            candidate.specificity,
            candidate.score,
        )

    def _adjudication_score(self, candidate: CandidateScore) -> float:
        score = candidate.score
        if (
            candidate.required_met
            and not candidate.hard_contradiction
            and candidate.matched_evidence
            and self._is_etiology_priority_candidate(candidate)
        ):
            score += self.etiology_priority_bonus
        elif (
            candidate.required_gaps
            and not candidate.hard_contradiction
            and candidate.matched_evidence
            and self._is_etiology_priority_candidate(candidate)
            and (
                candidate.coverage_score >= self.evidence_gap_coverage_threshold
                or candidate.residual_score <= self.evidence_gap_residual_threshold
            )
        ):
            score += self.etiology_priority_bonus * 0.5
        return round(min(1.0, score), 4)

    def _diagnosis_type_rank(self, candidate: CandidateScore) -> int:
        dtype = candidate.diagnosis_type.lower()
        if dtype in {"etiology", "metabolic", "structural"}:
            return 4
        if candidate.specificity >= 0.85:
            return 3
        if dtype == "disease":
            return 2
        if dtype in {"syndrome", "state", "complication"}:
            return 1
        return 0

    @staticmethod
    def _is_etiology_priority_candidate(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return (
            dtype in {"etiology", "metabolic", "structural"}
            or candidate.specificity >= 0.85
        )

    def render_candidate_table(self, decision: DiagnosisDecision, limit: int = 8) -> str:
        lines = ["【证据评分候选】"]
        for item in decision.candidates[:limit]:
            lines.append(
                f"- {item.diagnosis}: score={item.score:.3f}, support={item.support_score:.3f}, "
                f"prior={item.source_prior:.3f}, coverage={item.coverage_score:.3f}, "
                f"residual={item.residual_score:.3f}, required={item.required_met}, "
                f"hard_contradiction={item.hard_contradiction}, gaps={item.required_gaps[:3]}, "
                f"residual_evidence={item.residual_evidence[:4]}, evidence={item.matched_evidence[:5]}"
            )
        resolved = [
            f"{item.get('raw_name')}→{item.get('canonical_name')}({item.get('method')})"
            for item in decision.name_resolutions
            if item.get("canonical_name")
        ]
        if resolved:
            lines.append("【开放候选标准化】" + "；".join(resolved[:limit]))
        if decision.unresolved_candidates:
            lines.append("【未解析候选】" + "、".join(decision.unresolved_candidates[:limit]))
        return "\n".join(lines)

    @staticmethod
    def _clear_submission_marks(scores: Sequence[CandidateScore]) -> None:
        for item in scores:
            item.differential_only = False
            item.differential_only_reason = ""

    def differential_only_details(
        self,
        scores: Sequence[CandidateScore],
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for item in scores:
            if not item.differential_only:
                continue
            details.append(
                {
                    "diagnosis": item.diagnosis,
                    "reason": item.differential_only_reason,
                    "score": item.score,
                    "coverage_score": item.coverage_score,
                    "residual_score": item.residual_score,
                    "causal_relation_to_selected": item.causal_relation_to_selected,
                }
            )
        return details

    def differential_only_reasoning(
        self,
        scores: Sequence[CandidateScore],
        limit: int = 3,
    ) -> str:
        details = self.differential_only_details(scores)
        if not details:
            return ""
        parts = [
            f"{item['diagnosis']}：{item['reason']}"
            for item in details[:limit]
        ]
        return "仅鉴别诊断不提交：" + "；".join(parts) + "。"

    def resolve_open_candidates(self, result: Any) -> List[DiagnosisResolution]:
        return self.resolver.resolve_result(result)

    def _score_entry(
        self,
        entry: Dict[str, Any],
        prior: float,
        evidence: EvidenceBundle,
        candidate_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> CandidateScore:
        support_specs = list(entry.get("supporting_evidence", []) or [])
        matched: List[str] = []
        matched_weight = 0.0
        for spec in support_specs:
            hits = _matching_observations(spec, evidence.observations, polarity="positive")
            if not hits:
                continue
            weight = float(spec.get("weight", 0.2) or 0.2)
            best_confidence = max(item.confidence for item in hits)
            matched_weight += weight * best_confidence
            matched.extend(item.finding for item in hits)

        support_score = min(1.0, matched_weight)
        required_groups = entry.get("required_groups", []) or []
        required_gaps: List[str] = []
        for group in required_groups:
            if not isinstance(group, list) or not group:
                continue
            matched_group = any(
                _matching_observations(_coerce_spec(spec), evidence.observations, polarity="positive")
                for spec in group
            )
            if not matched_group:
                required_gaps.append(_render_required_group(group))
        required_met = not required_gaps

        contradicted: List[str] = []
        contradiction_penalty = 0.0
        hard_contradiction = False
        direct_negative = [
            item for item in evidence.observations
            if item.finding == f"diagnosis:{entry['name']}" and item.polarity == "negative"
        ]
        if direct_negative:
            contradicted.append(f"diagnosis:{entry['name']}")
            contradiction_penalty += 0.65
            hard_contradiction = True

        for spec in entry.get("contradictions", []) or []:
            spec = _coerce_spec(spec)
            hits = _matching_observations(spec, evidence.observations, polarity=spec.get("polarity", "positive"))
            if not hits:
                continue
            contradicted.extend(item.finding for item in hits)
            contradiction_penalty += float(spec.get("penalty", 0.35) or 0.35)
            hard_contradiction = hard_contradiction or bool(spec.get("hard", False))

        # Negative evidence against a required/supporting finding lowers the score even
        # when the knowledge entry did not repeat it as an explicit contradiction.
        supporting_findings = {str(spec.get("finding")) for spec in support_specs if spec.get("finding")}
        for item in evidence.observations:
            if item.polarity == "negative" and item.finding in supporting_findings:
                contradiction_penalty += 0.2
                contradicted.append(item.finding)

        coverage_score, residual_score, residual_evidence = self._explainability(
            support_specs,
            evidence,
            bool(matched),
        )
        explanation = coverage_score
        specificity = float(entry.get("specificity", 0.5) or 0.5)
        has_signal = bool(matched or prior > 0)
        specificity_score = specificity if has_signal else 0.0
        exam_match = self._expected_exam_match(entry, evidence) if has_signal else 0.0
        temporal = self._temporal_consistency(matched, evidence) if has_signal else 0.0
        age = self._age_match(entry, evidence) if has_signal else 0.0
        risk = self._risk_factor_match(entry, evidence) if has_signal else 0.0
        objective_evidence = self._has_objective_evidence(matched, evidence)
        gap_penalty = 0.0
        if required_gaps:
            if objective_evidence:
                gap_penalty = min(0.12, 0.04 * len(required_gaps))
            else:
                gap_penalty = min(0.28, 0.18 + 0.05 * (len(required_gaps) - 1))
        raw_score = (
            self.weights["evidence"] * support_score
            + self.weights["prior"] * max(0.0, min(1.0, prior))
            + self.weights["specificity"] * specificity_score
            + self.weights["explain"] * explanation
            + self.weights["exam_match"] * exam_match
            + self.weights["temporal"] * temporal
            + self.weights["age"] * age
            + self.weights["risk"] * risk
            - self.weights["residual"] * residual_score
            - self.weights["contradiction"] * contradiction_penalty
            - gap_penalty
        )
        if not required_met and self.required_group_policy != "gap_only":
            raw_score = min(raw_score, self.differential_threshold - 0.01)
        score = max(0.0, min(1.0, raw_score))
        return CandidateScore(
            diagnosis=entry["name"],
            score=round(score, 4),
            support_score=round(support_score, 4),
            source_prior=round(prior, 4),
            explanation_score=round(explanation, 4),
            coverage_score=round(coverage_score, 4),
            residual_score=round(residual_score, 4),
            contradiction_penalty=round(contradiction_penalty, 4),
            required_met=required_met,
            hard_contradiction=hard_contradiction,
            matched_evidence=list(dict.fromkeys(matched)),
            contradicted_evidence=list(dict.fromkeys(contradicted)),
            required_gaps=list(dict.fromkeys(required_gaps)),
            residual_evidence=list(dict.fromkeys(residual_evidence))[:12],
            component_scores={
                "evidence": round(support_score, 4),
                "prior": round(max(0.0, min(1.0, prior)), 4),
                "specificity": round(specificity_score, 4),
                "explain": round(explanation, 4),
                "coverage": round(coverage_score, 4),
                "residual": round(residual_score, 4),
                "exam_match": round(exam_match, 4),
                "temporal": round(temporal, 4),
                "age": round(age, 4),
                "risk": round(risk, 4),
                "contradiction": round(contradiction_penalty, 4),
                "required_gap_penalty": round(gap_penalty, 4),
                "objective_evidence": 1.0 if objective_evidence else 0.0,
            },
            candidate_sources=list(candidate_sources or []),
            diagnosis_type=str(entry.get("diagnosis_type") or "disease"),
            parent_diagnosis=str(entry.get("parent_diagnosis") or ""),
            specificity=specificity,
        )

    def _explainability(
        self,
        support_specs: Sequence[Dict[str, Any]],
        evidence: EvidenceBundle,
        has_matched_signal: bool,
    ) -> Tuple[float, float, List[str]]:
        major = evidence.major()
        if not major:
            coverage = 0.5 if has_matched_signal else 0.0
            return coverage, 0.0, []

        total_weight = 0.0
        explained_weight = 0.0
        residual: List[str] = []
        for observation in major:
            weight = self._observation_explainability_weight(observation)
            total_weight += weight
            if any(_observation_matches(spec, observation) for spec in support_specs):
                explained_weight += weight
            else:
                residual.append(observation.finding)

        if total_weight <= 0:
            return 0.0, 0.0, []
        coverage = max(0.0, min(1.0, explained_weight / total_weight))
        residual_score = max(0.0, min(1.0, 1.0 - coverage))
        return coverage, residual_score, residual

    @staticmethod
    def _observation_explainability_weight(observation: Observation) -> float:
        if observation.finding.startswith("diagnosis:"):
            return 1.15
        if observation.value is not None or observation.direction:
            return 1.05
        if observation.source != "问诊":
            return 1.0
        if observation.finding.startswith("symptom:"):
            return 0.65
        return 0.8

    def _expected_exam_match(self, entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        exams = [str(item).strip() for item in entry.get("discriminating_exams", []) or [] if str(item).strip()]
        if not exams:
            return 0.0
        sources = {_compact_text(item.source) for item in evidence.observations if item.source}
        if not sources:
            return 0.0
        matched = 0
        for exam in exams:
            compact_exam = _compact_text(exam)
            if any(compact_exam in source or source in compact_exam for source in sources):
                matched += 1
        denominator = max(1, min(3, len(exams)))
        return min(1.0, matched / denominator)

    @staticmethod
    def _has_objective_evidence(matched: Sequence[str], evidence: EvidenceBundle) -> bool:
        matched_set = set(matched or [])
        if not matched_set:
            return False
        for item in evidence.observations:
            if item.finding not in matched_set or item.polarity != "positive":
                continue
            if item.source != "问诊":
                return True
            if item.finding.startswith("diagnosis:"):
                return True
            if item.finding not in {"palpitation", "muscle_cramp", "dyspnea", "weakness", "dizziness"} and not item.finding.startswith("symptom:"):
                return True
        return False

    @staticmethod
    def _temporal_consistency(matched: Sequence[str], evidence: EvidenceBundle) -> float:
        if not matched:
            return 0.0
        matched_set = set(matched)
        observations = [
            item for item in evidence.observations
            if item.finding in matched_set and item.polarity == "positive"
        ]
        if not observations:
            return 0.0
        if any(item.temporality for item in observations):
            return 0.7
        return 0.4

    @staticmethod
    def _age_match(entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        diagnosis = str(entry.get("name") or "")
        department = str(entry.get("department") or "")
        age_values = [
            item.value for item in evidence.observations
            if item.finding == "field:age" and item.value is not None
        ]
        if not age_values:
            return 0.0
        age = age_values[0]
        if age < 18 and ("儿" in department or diagnosis in {"先天性心脏病", "房间隔缺损"}):
            return 1.0
        if age >= 60 and diagnosis in {"骨质疏松症", "冠心病", "终末期肾病"}:
            return 0.7
        if age < 18 and diagnosis in {"骨质疏松症", "冠心病"}:
            return 0.0
        return 0.35

    @staticmethod
    def _risk_factor_match(entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        diagnosis = str(entry.get("name") or "")
        findings = set(evidence.findings("positive"))
        disease_signals = {
            "先天性心脏病": {
                "cyanosis", "feeding_diaphoresis", "congenital_heart_defect",
                "ventricular_septal_defect", "right_to_left_shunt", "pulmonary_hypertension",
            },
            "肺动脉瓣狭窄": {"pulmonary_valve_gradient", "pulmonary_valve_stenosis", "cyanosis"},
            "肺不张": {"atelectasis", "choking_event", "aspiration_risk"},
            "终末期肾病": {"renal_impairment", "egfr_low", "urea_elevated", "oliguria", "hyperkalemia"},
            "卵巢过度刺激综合征": {
                "ohss_risk", "ovarian_enlargement", "ascites", "hemoconcentration", "hypoalbuminemia",
            },
            "门静脉高压": {"portal_vein_dilation", "portal_flow_abnormal", "splenomegaly", "ascites", "varices"},
        }
        expected = disease_signals.get(diagnosis, set())
        if not expected:
            return 0.0
        overlap = len(expected & findings)
        return min(1.0, overlap / max(1, min(3, len(expected))))

    def _candidate_priors(
        self,
        llm_result: Dict[str, Any],
        rag_chunks: Sequence[Dict[str, Any]],
        resolutions: Optional[Sequence[DiagnosisResolution]] = None,
    ) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        resolved_items = list(resolutions or self.resolver.resolve_result(llm_result))
        for index, item in enumerate(resolved_items):
            name = item.canonical_name
            if not name:
                continue
            rank_prior = max(0.65, 1.0 - index * 0.10)
            mapping_confidence = item.confidence * item.model_confidence
            prior = rank_prior * mapping_confidence
            priors[name] = max(priors.get(name, 0.0), prior)

        for chunk in rag_chunks:
            if chunk.get("type") != "disease_profile":
                continue
            resolution = self.resolver.resolve(chunk.get("title"))
            name = resolution.canonical_name
            if not name:
                continue
            try:
                score = float(chunk.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            priors[name] = max(priors.get(name, 0.0), score)
        return priors

    def _select_final(self, pool: Sequence[CandidateScore]) -> List[CandidateScore]:
        selected: List[CandidateScore] = []
        suppressed: Set[str] = set()
        for candidate in pool:
            if not candidate.trusted:
                continue
            if candidate.diagnosis in suppressed:
                continue
            entry = self.knowledge.get(candidate.diagnosis)
            parent = candidate.parent_diagnosis
            if parent:
                selected = [
                    item for item in selected
                    if item.diagnosis != parent or self._has_independent_state_evidence(item)
                ]
            # If a more specific selected child already points to this candidate as parent,
            # retain the parent only when it has independent state evidence.
            if any(item.parent_diagnosis == candidate.diagnosis for item in selected):
                if not self._has_independent_state_evidence(candidate):
                    self._mark_differential_only(
                        candidate,
                        "作为更泛化的父诊断保留鉴别，但已有更具体诊断且缺少独立状态证据，不作为最终诊断提交。",
                    )
                    continue
            if self._is_explained_secondary_manifestation(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    "作为已选病因/结构诊断可解释的表现或并发状态保留鉴别，缺少独立客观证据，不作为最终诊断提交。",
                )
                continue
            if self._is_unrelated_high_residual(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    self._differential_only_reason(candidate, selected),
                )
                continue
            if not self._is_final_companion_eligible(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    self._differential_only_reason(candidate, selected),
                )
                continue
            selected.append(candidate)
            suppressed.update(str(item) for item in entry.get("suppress_diagnoses", []) or [])
            if len(selected) >= self.max_final_diagnoses:
                break
        return selected

    def _append_independent_states(
        self,
        selected: Sequence[CandidateScore],
        scores: Sequence[CandidateScore],
    ) -> List[CandidateScore]:
        result = list(selected)
        if len(result) >= self.max_final_diagnoses:
            return result
        if not any(self._is_causal_primary(item) for item in result):
            return result
        selected_names = {item.diagnosis for item in result}
        for candidate in scores:
            if candidate.diagnosis in selected_names:
                continue
            if not candidate.trusted:
                continue
            if candidate.hard_contradiction or not candidate.matched_evidence:
                continue
            if candidate.score < self.differential_threshold:
                continue
            if self._is_unrelated_high_residual(candidate, result):
                continue
            if not self._is_final_companion_eligible(candidate, result):
                continue
            if not self._has_independent_state_evidence(candidate):
                continue
            result.append(candidate)
            selected_names.add(candidate.diagnosis)
            if len(result) >= self.max_final_diagnoses:
                break
        return result

    def _is_final_companion_eligible(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return True
        if not candidate.trusted or candidate.hard_contradiction:
            return False
        if self._is_related_to_selected(candidate, selected):
            if self._is_secondary_manifestation(candidate):
                return self._has_independent_state_evidence(candidate)
            return True
        if (
            self._is_secondary_manifestation(candidate)
            and self._has_independent_state_evidence(candidate)
            and any(self._diagnoses_submission_related(candidate.diagnosis, item.diagnosis) for item in selected)
        ):
            return True
        return self._explains_selected_residual(candidate, selected)

    def _is_related_to_selected(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        return any(
            self._diagnoses_submission_related(candidate.diagnosis, item.diagnosis)
            for item in selected
        )

    def _explains_selected_residual(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        residual = set()
        for item in selected:
            residual.update(item.residual_evidence)
        if not residual:
            return False
        if not residual.intersection(candidate.matched_evidence):
            return False
        best_coverage = max(item.coverage_score for item in selected)
        best_residual = min(item.residual_score for item in selected)
        coverage_ok = candidate.coverage_score + 0.18 >= best_coverage
        residual_ok = candidate.residual_score <= best_residual + 0.18
        return coverage_ok and residual_ok

    def _mark_differential_only(self, candidate: CandidateScore, reason: str) -> None:
        candidate.differential_only = True
        candidate.differential_only_reason = reason

    def _differential_only_reason(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> str:
        primary = selected[0] if selected else None
        if not primary:
            return "作为鉴别保留，但未满足最终提交条件。"
        selected_residual = list(dict.fromkeys(
            item for diagnosis in selected for item in diagnosis.residual_evidence
        ))
        unexplained = [
            item for item in selected_residual
            if item not in set(candidate.matched_evidence)
        ][:3]
        evidence_gap = ""
        if "癌" in candidate.diagnosis or "肿瘤" in candidate.diagnosis:
            tumor_specific = any(
                token in evidence
                for evidence in candidate.matched_evidence
                for token in ("肿瘤", "癌", "占位", "结节", "肿块", "tumor")
            )
            if not tumor_specific and f"diagnosis:{candidate.diagnosis}" not in candidate.matched_evidence:
                evidence_gap = "且缺少肿瘤特异证据，"
        residual_text = "、".join(unexplained) if unexplained else "主诊断残余核心证据"
        return (
            f"作为鉴别保留，但相对{primary.diagnosis}解释力不足"
            f"（coverage {candidate.coverage_score:.2f} vs {primary.coverage_score:.2f}，"
            f"residual {candidate.residual_score:.2f} vs {primary.residual_score:.2f}），"
            f"未解释{residual_text}，{evidence_gap}且无明确因果/并发关系，不作为最终诊断提交。"
        )

    def _is_explained_secondary_manifestation(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        if any(self._diagnosis_causes(item.diagnosis, candidate.diagnosis) for item in selected):
            return not self._has_independent_state_evidence(candidate)
        if not self._is_secondary_manifestation(candidate):
            return False
        if not any(self._is_causal_primary(item) for item in selected):
            return False
        return not self._has_independent_state_evidence(candidate)

    def _is_unrelated_high_residual(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        if candidate.residual_score < self.residual_drop_threshold:
            return False
        if candidate.coverage_score >= self.evidence_gap_coverage_threshold:
            return False
        if self._is_secondary_manifestation(candidate) and self._has_independent_state_evidence(candidate):
            return False
        return not any(
            self._diagnoses_causally_related(candidate.diagnosis, item.diagnosis)
            for item in selected
        )

    @staticmethod
    def _is_secondary_manifestation(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return (
            dtype in {"syndrome", "state", "complication"}
            or candidate.diagnosis in _SECONDARY_MANIFESTATION_DIAGNOSES
        )

    @staticmethod
    def _is_causal_primary(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return dtype in {"etiology", "metabolic", "structural"} or candidate.specificity >= 0.85

    @staticmethod
    def _has_independent_state_evidence(candidate: CandidateScore) -> bool:
        if candidate.diagnosis == "心律失常":
            return any(
                finding in candidate.matched_evidence
                for finding in ("symptom:晕厥", "symptom:低血压", "symptom:持续心动过速")
            )
        if f"diagnosis:{candidate.diagnosis}" in candidate.matched_evidence:
            return True
        return (
            candidate.diagnosis == "心力衰竭"
            and "heart_failure_state" in candidate.matched_evidence
        )

    def _diagnosis_causes(self, cause: str, effect: str) -> bool:
        cause_entry = self.knowledge.get(cause)
        effect_entry = self.knowledge.get(effect)
        return (
            effect in set(str(item) for item in cause_entry.get("causes", []) or [])
            or cause in set(str(item) for item in effect_entry.get("caused_by", []) or [])
        )

    def _diagnoses_causally_related(self, left: str, right: str) -> bool:
        if left == right:
            return True
        return self._diagnosis_causes(left, right) or self._diagnosis_causes(right, left)

    def _diagnoses_submission_related(self, left: str, right: str) -> bool:
        if self._diagnoses_causally_related(left, right):
            return True
        left_entry = self.knowledge.get(left)
        right_entry = self.knowledge.get(right)
        left_parent = str(left_entry.get("parent_diagnosis") or "")
        right_parent = str(right_entry.get("parent_diagnosis") or "")
        return (
            bool(left_parent and left_parent == right)
            or bool(right_parent and right_parent == left)
            or bool(left_parent and right_parent and left_parent == right_parent)
        )

    def _annotate_causal_relations(
        self,
        scores: Sequence[CandidateScore],
        selected: Sequence[CandidateScore],
    ) -> None:
        selected_names = [item.diagnosis for item in selected]
        for candidate in scores:
            if candidate.diagnosis in selected_names:
                candidate.causal_relation_to_selected = "selected"
                continue
            relation = ""
            for name in selected_names:
                if self._diagnosis_causes(name, candidate.diagnosis):
                    relation = f"caused_by:{name}"
                    break
                if self._diagnosis_causes(candidate.diagnosis, name):
                    relation = f"causes:{name}"
                    break
            candidate.causal_relation_to_selected = relation or (
                "unrelated_to_selected" if selected_names else ""
            )

    def _fallback_candidate(
        self,
        priors: Dict[str, float],
        scores: Sequence[CandidateScore],
    ) -> Optional[CandidateScore]:
        for name, _ in sorted(priors.items(), key=lambda item: item[1], reverse=True):
            for score in scores:
                if (
                    score.diagnosis == name
                    and not score.hard_contradiction
                    and score.matched_evidence
                    and score.trusted
                ):
                    return score
        # Do not silently turn an all-zero evidence set into whichever disease
        # happens to be first in the catalog. The caller can ask for more data
        # or submit a controlled low-confidence candidate supplied by LLM/RAG.
        return None

    @staticmethod
    def _reasoning(selected: Sequence[CandidateScore], unexplained: Sequence[str]) -> str:
        if not selected:
            return "证据裁决未找到可提交的标准诊断。"
        parts = []
        for item in selected:
            evidence = "、".join(item.matched_evidence[:6]) or "有限临床证据"
            parts.append(
                f"{item.diagnosis}由{evidence}支持"
                f"（证据评分{item.score:.2f}，解释覆盖{item.coverage_score:.2f}，残余{item.residual_score:.2f}）"
            )
        text = "证据裁决：" + "；".join(parts) + "。"
        if unexplained:
            text += "仍需关注未完全解释的证据：" + "、".join(unexplained[:6]) + "。"
        return text


def _matching_observations(
    spec: Dict[str, Any],
    observations: Sequence[Observation],
    polarity: str = "positive",
) -> List[Observation]:
    return [
        item for item in observations
        if item.polarity == polarity and _observation_matches(spec, item)
    ]


def _observation_matches(spec: Dict[str, Any], item: Observation) -> bool:
    finding = str(spec.get("finding") or "")
    if finding and finding != item.finding:
        return False
    direction = str(spec.get("direction") or "")
    if direction and direction != item.direction:
        return False
    source_contains = str(spec.get("source_contains") or "")
    if source_contains and source_contains.lower() not in item.source.lower():
        return False
    terms = spec.get("terms") or []
    if isinstance(terms, str):
        terms = [terms]
    if terms and not any(str(term).lower() in item.raw_text.lower() for term in terms):
        return False
    if spec.get("min_value") is not None:
        if item.value is None or item.value < float(spec["min_value"]):
            return False
    if spec.get("max_value") is not None:
        if item.value is None or item.value > float(spec["max_value"]):
            return False
    return bool(finding or direction or source_contains or terms or spec.get("min_value") is not None or spec.get("max_value") is not None)


def _coerce_spec(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"finding": str(value)}


def _render_required_group(group: Sequence[Any]) -> str:
    labels: List[str] = []
    for item in group:
        spec = _coerce_spec(item)
        label = str(spec.get("finding") or "")
        if not label and spec.get("terms"):
            terms = spec.get("terms")
            if isinstance(terms, str):
                terms = [terms]
            label = "/".join(str(term) for term in terms[:3])
        if not label and spec.get("source_contains"):
            label = f"source:{spec.get('source_contains')}"
        if label:
            labels.append(label)
    return "|".join(labels) or "required_evidence"


def _compact_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _dedupe_specs(items: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        spec = _coerce_spec(item)
        key = json.dumps(spec, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(spec)
    return result


def _dedupe_objects(items: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default
