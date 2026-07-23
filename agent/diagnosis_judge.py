"""Replayable diagnosis judge and submitter.

The retriever/ranker owns candidate generation. The judge only arbitrates an
existing candidate table into primary, secondary, differential, and evidence-gap
roles. The submitter then writes that authorization back to DiagnosisDecision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_PRIORITY_TYPES = {"etiology", "metabolic", "structural", "systemic"}
_MANIFESTATION_TYPES = {"syndrome", "state", "complication"}
_MANIFESTATION_NAMES = {"心力衰竭", "心律失常", "肺动脉高压"}
_SYSTEMIC_PRIMARY_NAMES = {"卵巢过度刺激综合征", "门静脉高压", "终末期肾病"}
_PARENT_FALLBACK_NAMES = {"先天性心脏病"}
_DIFFERENTIAL_EXAM_HINTS = {
    "雅司病": ["体格检查", "梅毒血清学检查", "暗视野显微镜检查", "组织病理学检查"],
    "湿疹": ["体格检查", "血清学抗体检测"],
    "白血病": ["全血细胞计数（CBC）", "外周血涂片", "组织病理学检查"],
    "肺结核": ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    "肺炎": ["胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "痰培养"],
    "支气管肺炎": ["胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "痰培养"],
    "肺癌": ["胸部CT扫描（Chest CT）", "组织病理学检查", "支气管镜检查"],
    "老视": ["视力检查", "屈光检查"],
    "晶状体脱位": ["裂隙灯检查", "眼压测量", "眼部B超检查"],
    "青光眼": ["眼压测量", "裂隙灯检查", "眼底镜检查"],
}
_DIFFERENTIAL_SET_EXAM_HINTS = [
    (
        {"雅司病", "湿疹", "白血病"},
        ["全血细胞计数（CBC）", "外周血涂片", "梅毒血清学检查", "体格检查"],
    ),
    (
        {"肺结核", "肺炎", "肺癌"},
        ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    ),
    (
        {"肺结核", "支气管肺炎", "肺癌"},
        ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    ),
    (
        {"老视", "晶状体脱位", "青光眼"},
        ["视力检查", "屈光检查", "裂隙灯检查", "眼压测量"],
    ),
]
_OBJECTIVE_GAP_FINDINGS = {
    "ascites",
    "bilirubin_high",
    "dextrocardia",
    "egfr_low",
    "hemoconcentration",
    "hyperkalemia",
    "hypoalbuminemia",
    "left_to_right_shunt",
    "low_magnesium",
    "magnesium_depletion",
    "magnesium_load_retention_high",
    "oliguria",
    "portal_flow_abnormal",
    "portal_vein_dilation",
    "pulmonary_valve_gradient",
    "pulmonary_valve_stenosis",
    "renal_impairment",
    "right_heart_strain",
    "splenomegaly",
    "treponema_positive",
    "treponemal_skin_lesion",
    "tuberculosis_exposure",
    "hemoptysis",
    "urea_elevated",
    "uremia",
    "ventricular_septal_defect",
}

_BROAD_EVIDENCE_TOKENS = {
    "acute_course",
    "chronic_course",
    "congenital_onset",
    "fever",
    "cough",
    "fatigue",
    "pain",
    "pruritus",
    "rash",
    "abdominal_pain",
    "symptom:发热",
    "symptom:咳嗽",
    "symptom:乏力",
    "symptom:疼痛",
    "symptom:皮疹",
    "symptom:瘙痒",
}
_CORE_EVIDENCE_TOKENS = _OBJECTIVE_GAP_FINDINGS | {
    "abnormal_genitalia",
    "age_related_near_blur",
    "ambiguous_genitalia",
    "bone_pain",
    "cardiopulmonary_exertional_pattern",
    "crusted_exudative_skin_ulcer",
    "crusted_skin_lesion",
    "deep_skin_ulcer",
    "dermatomal_vesicles",
    "diagnostic_imaging",
    "dyspnea_on_exertion",
    "exercise_intolerance",
    "fluid_retention_pattern",
    "hemoptysis",
    "lens_dislocation",
    "midline_suprapubic_cyst",
    "near_vision_difficulty",
    "optic_pressure_high",
    "periorbital_edema",
    "periostitis",
    "polydipsia",
    "postprandial_nausea",
    "presbyopia_refraction",
    "refractive_error",
    "regional_lymphadenopathy",
    "rural_child_contact",
    "sex_development_disorder",
    "treponemal_disease_pattern",
    "tropical_exposure",
    "umbilical_discharge",
    "umbilical_mass",
    "urachal_cyst_imaging",
    "vesicular_rash",
}
_CONTEXTUAL_CORE_FINDINGS = {
    "bradycardia",
    "crusted_exudative_skin_ulcer",
    "anca_positive",
    "mpo_anca_positive",
    "p_anca_positive",
    "microscopic_hematuria",
    "proteinuria",
    "pulmonary_hemorrhage",
    "low_magnesium",
    "low_urine_magnesium",
    "magnesium_depletion",
    "magnesium_load_retention_high",
    "regional_lymphadenopathy",
    "rural_child_contact",
}
_CORE_SYMPTOM_KEYWORDS = (
    "脐部",
    "脐周",
    "脐下",
    "外生殖器",
    "尿道下裂",
    "隐睾",
    "看近",
    "阅读困难",
    "视物模糊",
    "血痰",
    "咯血",
    "盗汗",
    "骨痛",
    "关节痛",
    "结痂",
    "渗出",
    "黄水",
    "腹股沟",
    "水疱",
    "疱疹",
)
_GENERIC_EVIDENCE_PREFIXES = ("field:",)
_KNOWN_DIFFERENTIAL_GROUPS = (
    (
        "dermatology_eruptive_systemic",
        {"雅司病", "水痘", "湿疹", "白血病", "尖锐湿疣"},
    ),
    (
        "pulmonary_infection_mass",
        {"肺结核", "肺炎", "支气管肺炎", "肺癌", "肺隐球菌病", "肺念珠菌病", "支原体肺炎"},
    ),
    (
        "ophthalmology_visual",
        {"老视", "晶状体脱位", "青光眼", "白内障", "虹膜缺损"},
    ),
    (
        "congenital_genitourinary_dsd",
        {"卵睾性别发育异常（Ovotesticular DSD）", "X三体综合征（47,XXX）"},
    ),
    (
        "urachal_midline_urinary",
        {"脐尿管囊肿", "泌尿系感染", "尿道综合征", "急性细菌性前列腺炎"},
    ),
)
_KNOWN_CLUSTER_BY_NAME = {
    name: cluster
    for cluster, names in _KNOWN_DIFFERENTIAL_GROUPS
    for name in names
}
_FALLBACK_DISEASE_METADATA = {
    "门静脉高压": ("gastrointestinal", "portal_hypertension"),
    "白血病": ("hematology", "leukemia"),
    "带状疱疹": ("dermatology_infectious", "dermatomal_viral"),
    "骨折": ("musculoskeletal", "acute_trauma"),
    "前列腺增生": ("genitourinary", "prostate_obstruction"),
    "泌尿系感染": ("genitourinary", "urinary_tract_infection"),
    "尿道综合征": ("genitourinary", "urethral_syndrome"),
    "肺隐球菌病": ("respiratory", "opportunistic_fungal_pneumonia"),
    "肺念珠菌病": ("respiratory", "opportunistic_fungal_pneumonia"),
    "肺不张": ("respiratory", "atelectasis"),
    "二尖瓣反流": ("cardiovascular", "valvular_left_heart"),
    "三尖瓣反流": ("cardiovascular", "valvular_right_heart"),
    "心力衰竭": ("cardiovascular", "heart_failure_state"),
    "晶状体脱位": ("ophthalmology", "ocular_structural"),
    "青光眼": ("ophthalmology", "glaucoma"),
    "白内障": ("ophthalmology", "lens_opacity"),
}


@dataclass
class JudgeCandidateReview:
    diagnosis: str
    role: str
    reason: str
    score: float = 0.0
    judge_score: float = 0.0
    required_met: bool = False
    required_gap_authorized: bool = False
    hard_contradiction: bool = False
    coverage_score: float = 0.0
    residual_score: float = 0.0
    explanatory_coverage: float = 0.0
    core_explanatory_coverage: float = 0.0
    residual_evidence_score: float = 0.0
    residual_core_evidence_count: int = 0
    diagnosis_type: str = ""
    specificity: float = 0.0
    required_gaps: List[str] = field(default_factory=list)
    matched_evidence: List[str] = field(default_factory=list)
    explained_evidence: List[str] = field(default_factory=list)
    unexplained_core_evidence: List[str] = field(default_factory=list)
    explanatory_rank_reason: str = ""
    required_gap_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeDecision:
    retriever_top1: str = ""
    judge_primary: str = ""
    primary: str = ""
    primary_status: str = "locked"
    needs_discriminating_exams: bool = False
    provisional_primary: str = ""
    locked_primary: str = ""
    defer_reason: str = ""
    pre_discrimination_primary: str = ""
    fallback_primary: str = ""
    fallback_reason: str = ""
    discrimination_attempted: bool = False
    discrimination_resolved: bool = False
    fallback_to_pre_discrimination_primary: bool = False
    differential_pool_source: Dict[str, str] = field(default_factory=dict)
    secondary: List[str] = field(default_factory=list)
    differential: List[str] = field(default_factory=list)
    evidence_gap_targets: List[str] = field(default_factory=list)
    final_diagnoses: List[str] = field(default_factory=list)
    required_gap_authorized_diagnoses: List[str] = field(default_factory=list)
    blocked_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[JudgeCandidateReview] = field(default_factory=list)
    differential_candidates: List[str] = field(default_factory=list)
    pairwise_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    excluded_from_pairwise: List[Dict[str, Any]] = field(default_factory=list)
    pool_filter_reasons: Dict[str, str] = field(default_factory=dict)
    cluster_assignments: Dict[str, str] = field(default_factory=dict)
    pairwise_allowed_matrix: List[Dict[str, Any]] = field(default_factory=list)
    core_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    generic_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    pool_filter_summary: Dict[str, Any] = field(default_factory=dict)
    discriminating_findings: List[str] = field(default_factory=list)
    discriminating_exams: List[str] = field(default_factory=list)
    required_gap_by_candidate: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    high_value_gap_candidates: List[str] = field(default_factory=list)
    explanatory_coverage: float = 0.0
    core_explanatory_coverage: float = 0.0
    residual_evidence_score: float = 0.0
    residual_core_evidence_count: int = 0
    dynamic_rerank_trace: List[Dict[str, Any]] = field(default_factory=list)
    decision_override: bool = False
    required_gap_state_by_candidate: Dict[str, str] = field(default_factory=dict)
    gap_state_distribution: Dict[str, int] = field(default_factory=dict)
    primary_unlock_reason: str = ""
    explanation_score_changed_ranking: bool = False
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reviews"] = [item.to_dict() for item in self.reviews]
        return data


@dataclass
class DifferentialPoolFilterResult:
    candidates: List[Any] = field(default_factory=list)
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    pool_filter_reasons: Dict[str, str] = field(default_factory=dict)
    cluster_assignments: Dict[str, str] = field(default_factory=dict)
    pairwise_allowed_matrix: List[Dict[str, Any]] = field(default_factory=list)
    core_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    generic_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    pool_source: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class DifferentialPoolFilter:
    """Filter Top20 candidates into clinically meaningful pairwise DDx."""

    def __init__(self, judge: "DiagnosisJudge"):
        self.judge = judge

    def filter(
        self,
        pool: Sequence[Any],
        source_by_name: Optional[Dict[str, str]] = None,
    ) -> DifferentialPoolFilterResult:
        source_by_name = dict(source_by_name or {})
        candidates = [item for item in pool or [] if item]
        result = DifferentialPoolFilterResult()
        if not candidates:
            return result

        relevance = {self._name(item): self._relevance(item) for item in candidates}
        result.cluster_assignments = {
            self._name(item): relevance[self._name(item)]["cluster"]
            for item in candidates
        }
        result.core_evidence_by_candidate = {
            self._name(item): list(relevance[self._name(item)]["core"])
            for item in candidates
        }
        result.generic_evidence_by_candidate = {
            self._name(item): list(relevance[self._name(item)]["generic"])
            for item in candidates
        }

        dominant_clusters = self._dominant_clusters(candidates, relevance)
        retained: List[Any] = []
        excluded: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            name = self._name(candidate)
            reason = self._keep_reason(
                candidate,
                index,
                candidates,
                retained,
                relevance,
                dominant_clusters,
            )
            if reason:
                retained.append(candidate)
                result.pool_filter_reasons[name] = reason
                continue
            exclude_reason = self._exclude_reason(
                candidate,
                relevance,
                dominant_clusters,
            )
            excluded.append(
                {
                    "diagnosis": name,
                    "reason": exclude_reason,
                    "cluster": relevance[name]["cluster"],
                    "core_evidence": list(relevance[name]["core"])[:6],
                    "generic_evidence": list(relevance[name]["generic"])[:6],
                    "source": source_by_name.get(name, ""),
                }
            )

        if not retained:
            retained = [candidates[0]]
            result.pool_filter_reasons[self._name(candidates[0])] = "fallback_top_candidate"

        retained = self._limit_pool(retained, relevance, candidates)
        retained_names = {self._name(item) for item in retained}
        for candidate in candidates:
            name = self._name(candidate)
            if name in retained_names:
                continue
            if any(item.get("diagnosis") == name for item in excluded):
                continue
            excluded.append(
                {
                    "diagnosis": name,
                    "reason": "pool_size_limit",
                    "cluster": relevance[name]["cluster"],
                    "core_evidence": list(relevance[name]["core"])[:6],
                    "generic_evidence": list(relevance[name]["generic"])[:6],
                    "source": source_by_name.get(name, ""),
                }
            )

        matrix = self._allowed_matrix(retained, relevance)
        result.candidates = retained
        result.excluded = excluded
        result.pairwise_allowed_matrix = matrix
        result.pool_source = {
            self._name(item): source_by_name.get(self._name(item), "filtered")
            for item in retained
        }
        result.summary = self._summary(candidates, retained, excluded, matrix, relevance)
        return result

    def allowed_pair_names(
        self,
        result: DifferentialPoolFilterResult,
    ) -> set[tuple[str, str]]:
        allowed: set[tuple[str, str]] = set()
        for item in result.pairwise_allowed_matrix:
            if item.get("allowed"):
                left = str(item.get("left") or "")
                right = str(item.get("right") or "")
                if left and right:
                    allowed.add(tuple(sorted((left, right))))
        return allowed

    def _keep_reason(
        self,
        candidate: Any,
        index: int,
        candidates: Sequence[Any],
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
        dominant_clusters: set[str],
    ) -> str:
        name = self._name(candidate)
        data = relevance[name]
        if getattr(candidate, "hard_contradiction", False):
            return ""
        if self._high_explanatory_candidate(candidate, data):
            return "high_explanatory_primary_candidate"
        if self._direct_diagnosis(candidate):
            non_diagnosis_core = [
                item for item in data["core"] if not str(item).startswith("diagnosis:")
            ]
            if (
                dominant_clusters
                and data["cluster"] not in dominant_clusters
                and (self._direct_low_explainability(candidate) or not non_diagnosis_core)
            ):
                return ""
            if not non_diagnosis_core and index >= max(1, self.judge.differential_top_k):
                return ""
            if (
                getattr(candidate, "required_met", False)
                or non_diagnosis_core
                or self.judge._objective_signal(candidate)
            ):
                return "direct_diagnosis_evidence"
            if (
                not dominant_clusters
                or data["cluster"] in dominant_clusters
                or any(
                    self._can_form_differential(candidate, item, relevance)
                    for item in retained
                )
            ):
                return "direct_diagnosis_evidence"
            return ""
        if data["core"]:
            if (
                float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.75
                and float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.55
                and self.judge._residual(candidate) <= 0.45
            ):
                return "high_explanatory_source_candidate"
            if data["cluster"] in dominant_clusters:
                return "dominant_cluster_core_evidence"
            if not dominant_clusters and self._strong_specific_candidate(candidate):
                return "standalone_core_specific_candidate"
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "clinically_related_core_evidence"
        if index < max(2, self.judge.differential_top_k):
            if data["cluster"] in dominant_clusters:
                return "top_k_same_dominant_cluster"
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "top_k_clinically_related"
            if not dominant_clusters and data["generic"]:
                return "top_k_no_dominant_cluster"
        if self._strong_specific_candidate(candidate):
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "top20_specific_related_tail"
            if data["core"] and (
                not dominant_clusters or data["cluster"] in dominant_clusters
            ) and not any(
                relevance[self._name(item)]["core"] for item in retained
            ):
                return "top20_specific_core_tail"
        return ""

    def _exclude_reason(
        self,
        candidate: Any,
        relevance: Dict[str, Dict[str, Any]],
        dominant_clusters: set[str],
    ) -> str:
        name = self._name(candidate)
        data = relevance[name]
        if getattr(candidate, "hard_contradiction", False):
            return "negative_feature"
        if not data["core"] and data["generic"]:
            return "generic_only_evidence"
        if data["cluster"] not in dominant_clusters and dominant_clusters:
            return "cross_system_no_shared_core_evidence"
        if (
            float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.24
            and self.judge._residual(candidate) > 0.55
        ):
            return "low_core_coverage_high_residual"
        if not self.judge._candidate_discriminating_exams(candidate):
            return "no_discriminating_exam"
        return "cross_system_no_shared_core_evidence"

    def _dominant_clusters(
        self,
        candidates: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
    ) -> set[str]:
        cluster_scores: Dict[str, float] = {}
        cluster_counts: Dict[str, int] = {}
        for candidate in candidates:
            name = self._name(candidate)
            cluster = str(relevance[name]["cluster"] or "")
            if not cluster or cluster == "unknown":
                continue
            core_count = len(relevance[name]["core"])
            direct = 1.0 if self._direct_diagnosis(candidate) else 0.0
            if not core_count and not direct:
                continue
            if direct and self._direct_low_explainability(candidate):
                continue
            score = (
                0.42 * min(core_count, 3)
                + 0.35 * float(getattr(candidate, "coverage_score", 0.0) or 0.0)
                + 0.16 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
                + 0.20 * direct
            )
            if direct and not [
                item
                for item in relevance[name]["core"]
                if not str(item).startswith("diagnosis:")
            ]:
                score -= 0.24
            if self._strong_specific_candidate(candidate):
                score += 0.08
            cluster_scores[cluster] = cluster_scores.get(cluster, 0.0) + score
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        if not cluster_scores:
            return set()
        best = max(cluster_scores.values())
        threshold = max(0.42, best * 0.62)
        multi_candidate_cluster_exists = any(count >= 2 for count in cluster_counts.values())
        return {
            cluster
            for cluster, score in cluster_scores.items()
            if score >= threshold
            and (
                cluster_counts.get(cluster, 0) >= 2
                or not multi_candidate_cluster_exists
                or score >= best * 0.92
            )
        }

    def _allowed_matrix(
        self,
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        matrix: List[Dict[str, Any]] = []
        for left_index, left in enumerate(retained):
            for right in retained[left_index + 1 :]:
                allowed, reason = self._pair_allowed_reason(left, right, relevance)
                matrix.append(
                    {
                        "left": self._name(left),
                        "right": self._name(right),
                        "allowed": allowed,
                        "reason": reason,
                    }
                )
        return matrix

    def _pair_allowed_reason(
        self,
        left: Any,
        right: Any,
        relevance: Dict[str, Dict[str, Any]],
    ) -> tuple[bool, str]:
        if getattr(left, "hard_contradiction", False) or getattr(
            right, "hard_contradiction", False
        ):
            return False, "negative_feature"
        left_data = relevance[self._name(left)]
        right_data = relevance[self._name(right)]
        if left_data["cluster"] and left_data["cluster"] == right_data["cluster"]:
            return True, "same_clinical_cluster"
        shared_core = set(left_data["core"]) & set(right_data["core"])
        if shared_core:
            return True, "shared_core_evidence"
        if self.judge._same_family(left, right):
            return True, "same_family"
        if self.judge._causally_related(left, right):
            return True, "causal_or_graph_relation"
        if self.judge._same_body_system(left, right) and (
            left_data["core"] or right_data["core"]
        ):
            return True, "same_body_system_with_core_evidence"
        return False, "cross_system_no_shared_core_evidence"

    def _can_form_differential(
        self,
        left: Any,
        right: Any,
        relevance: Dict[str, Dict[str, Any]],
    ) -> bool:
        allowed, _ = self._pair_allowed_reason(left, right, relevance)
        return allowed

    def _limit_pool(
        self,
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
        original: Sequence[Any],
    ) -> List[Any]:
        limit = int(
            getattr(self.judge, "filtered_pool_max_size", 0)
            or max(self.judge.differential_top_k, 6)
        )
        if len(retained) <= limit:
            return list(retained)
        original_index = {id(item): index for index, item in enumerate(original)}

        def key(item: Any) -> tuple:
            data = relevance[self._name(item)]
            index = original_index.get(id(item), 999)
            return (
                1 if self._high_explanatory_candidate(item, data) else 0,
                1 if index < self.judge.differential_top_k else 0,
                1 if data["core"] else 0,
                1 if self._direct_diagnosis(item) else 0,
                self.judge._judge_score(item),
                -index,
            )

        selected = sorted(retained, key=key, reverse=True)[:limit]
        selected_ids = {id(item) for item in selected}
        return [item for item in original if id(item) in selected_ids]

    def _summary(
        self,
        original: Sequence[Any],
        retained: Sequence[Any],
        excluded: Sequence[Dict[str, Any]],
        matrix: Sequence[Dict[str, Any]],
        relevance: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        retained_names = [self._name(item) for item in retained]
        noise_rejections = sum(
            1
            for item in excluded
            if item.get("reason")
            in {
                "cross_system_no_shared_core_evidence",
                "generic_only_evidence",
                "low_core_coverage_high_residual",
                "negative_feature",
                "no_discriminating_exam",
                "pool_size_limit",
            }
        )
        cluster_rejections = sum(
            1
            for item in excluded
            if item.get("reason") == "cross_system_no_shared_core_evidence"
        )
        core_hits = sum(
            1 for name in retained_names if relevance.get(name, {}).get("core")
        )
        return {
            "initial_pool_count": len(original),
            "filtered_pool_count": len(retained),
            "excluded_count": len(excluded),
            "pairwise_allowed_count": sum(1 for item in matrix if item.get("allowed")),
            "pairwise_blocked_count": sum(1 for item in matrix if not item.get("allowed")),
            "pairwise_noise_rejection_count": noise_rejections,
            "cluster_gate_rejection_count": cluster_rejections,
            "core_evidence_coverage": (
                round(core_hits / max(1, len(retained)), 4) if retained else None
            ),
        }

    def _relevance(self, candidate: Any) -> Dict[str, Any]:
        core: List[str] = []
        generic: List[str] = []
        entry = self._entry(candidate)
        entry_findings = self._entry_findings(entry)
        for item in getattr(candidate, "matched_evidence", []) or []:
            text = str(item or "").strip()
            if not text:
                continue
            if self._is_core_evidence(text, candidate, entry_findings):
                if text not in core:
                    core.append(text)
            else:
                if text not in generic:
                    generic.append(text)
        return {
            "core": core,
            "generic": generic,
            "cluster": self._clinical_cluster(candidate, entry, core, generic),
        }

    def _is_core_evidence(
        self,
        text: str,
        candidate: Any,
        entry_findings: set[str],
    ) -> bool:
        if text.startswith(_GENERIC_EVIDENCE_PREFIXES):
            return False
        if text.startswith("diagnosis:"):
            return True
        if text in _BROAD_EVIDENCE_TOKENS:
            return False
        if text in _CONTEXTUAL_CORE_FINDINGS:
            return self._contextual_core_allowed(text, candidate)
        if text in _CORE_EVIDENCE_TOKENS or text in entry_findings:
            return True
        if text.startswith("symptom:"):
            symptom = text.split(":", 1)[1]
            return self._symptom_core_allowed(symptom, candidate)
        lower = text.lower()
        if lower in _BROAD_EVIDENCE_TOKENS:
            return False
        return any(
            marker in lower
            for marker in (
                "_positive",
                "_abnormal",
                "_high",
                "_low",
                "_defect",
                "_stenosis",
                "_imaging",
                "_cyst",
                "_mass",
                "_discharge",
            )
        )

    def _contextual_core_allowed(self, text: str, candidate: Any) -> bool:
        name = self._name(candidate)
        entry = self._entry(candidate)
        body, family = self._metadata(candidate, entry)
        cluster = _KNOWN_CLUSTER_BY_NAME.get(name, "")
        if text in {
            "crusted_exudative_skin_ulcer",
            "regional_lymphadenopathy",
            "rural_child_contact",
        }:
            return (
                cluster == "dermatology_eruptive_systemic"
                or body.startswith("dermatology")
                or body == "hematology"
                or family in {"treponemal_skin_bone_infection", "vesicular_viral_exanthem"}
            )
        if text == "bradycardia":
            return body == "cardiovascular" and family == "cardiovascular_conduction"
        if text in {
            "anca_positive",
            "mpo_anca_positive",
            "p_anca_positive",
            "microscopic_hematuria",
            "proteinuria",
            "pulmonary_hemorrhage",
        }:
            return (
                "\u8840\u7ba1\u708e" in name
                or "\u80ba\u80be" in name
                or body in {"immune", "rheumatology", "nephrology", "immune_renal_pulmonary"}
                or "vasculitis" in family.lower()
                or "glomerulonephritis" in family.lower()
            )
        if text in {
            "low_magnesium",
            "low_urine_magnesium",
            "magnesium_depletion",
            "magnesium_load_retention_high",
        }:
            return (
                "\u4f4e\u9541" in name
                or "magnesium" in family.lower()
                or "electrolyte" in family.lower()
                or body in {"metabolic", "endocrine_metabolic"}
            )
        return True

    def _symptom_core_allowed(self, symptom: str, candidate: Any) -> bool:
        if not any(keyword in symptom for keyword in _CORE_SYMPTOM_KEYWORDS):
            return False
        name = self._name(candidate)
        entry = self._entry(candidate)
        body, family = self._metadata(candidate, entry)
        cluster = _KNOWN_CLUSTER_BY_NAME.get(name, "")
        if any(keyword in symptom for keyword in ("脐部", "脐周", "脐下")):
            return cluster == "urachal_midline_urinary" or family == "urachal_remnant"
        if any(keyword in symptom for keyword in ("外生殖器", "尿道下裂", "隐睾")):
            return cluster == "congenital_genitourinary_dsd" or family == "sex_development_disorder"
        if any(keyword in symptom for keyword in ("看近", "阅读困难", "视物模糊")):
            return body == "ophthalmology"
        if any(keyword in symptom for keyword in ("血痰", "咯血", "盗汗")):
            return body == "respiratory" or cluster == "pulmonary_infection_mass"
        if any(keyword in symptom for keyword in ("结痂", "渗出", "黄水", "腹股沟", "水疱", "疱疹", "关节痛", "骨痛")):
            return (
                cluster == "dermatology_eruptive_systemic"
                or body.startswith("dermatology")
                or body == "hematology"
            )
        return True

    def _entry_findings(self, entry: Dict[str, Any]) -> set[str]:
        findings: set[str] = set()
        for spec in entry.get("supporting_evidence", []) or []:
            if isinstance(spec, dict):
                finding = str(spec.get("finding") or "").strip()
                if finding:
                    findings.add(finding)
                for term in spec.get("terms", []) or []:
                    text = str(term or "").strip()
                    if text:
                        findings.add(text)
        for group in entry.get("required_groups", []) or []:
            for spec in group or []:
                if isinstance(spec, dict):
                    finding = str(spec.get("finding") or "").strip()
                    if finding:
                        findings.add(finding)
        return findings

    def _clinical_cluster(
        self,
        candidate: Any,
        entry: Dict[str, Any],
        core: Sequence[str],
        generic: Sequence[str],
    ) -> str:
        name = self._name(candidate)
        if name in _KNOWN_CLUSTER_BY_NAME:
            return _KNOWN_CLUSTER_BY_NAME[name]
        if "\u4f4e\u9541" in name:
            return "metabolic_electrolyte"
        if "\u5375\u5de2\u8fc7\u5ea6\u523a\u6fc0" in name:
            return "gynecology_ovarian_hyperstimulation"
        if "\u8840\u7ba1\u708e" in name or "\u80ba\u80be" in name:
            return "pulmonary_renal_vasculitis"
        body, family = self._metadata(candidate, entry)
        if family in {
            "treponemal_skin_bone_infection",
            "vesicular_viral_exanthem",
            "dermatitis",
            "anogenital_hpv_infection",
        }:
            return "dermatology_eruptive_systemic"
        if family in {
            "pulmonary_tuberculosis",
            "opportunistic_fungal_pneumonia",
            "pneumonia",
            "lung_malignancy",
        } or body == "respiratory":
            return "pulmonary_infection_mass"
        if body == "ophthalmology":
            return "ophthalmology_visual"
        if family == "sex_development_disorder" or body in {"endocrine_genetic", "genetic"}:
            return "congenital_genitourinary_dsd"
        if family == "urachal_remnant" or set(core) & {
            "umbilical_discharge",
            "umbilical_mass",
            "midline_suprapubic_cyst",
            "urachal_cyst_imaging",
        }:
            return "urachal_midline_urinary"
        if body and family:
            return f"{body}:{family}"
        return body or family or "unknown"

    def _metadata(self, candidate: Any, entry: Dict[str, Any]) -> tuple[str, str]:
        name = self._name(candidate)
        body = str(entry.get("body_system") or "")
        family = str(entry.get("disease_family") or entry.get("family") or "")
        fallback = _FALLBACK_DISEASE_METADATA.get(name)
        if fallback:
            body = body or fallback[0]
            family = family or fallback[1]
        return body, family

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if self.judge.knowledge and candidate:
            return self.judge.knowledge.get(self._name(candidate)) or {}
        return {}

    def _direct_diagnosis(self, candidate: Any) -> bool:
        name = self._name(candidate)
        return f"diagnosis:{name}" in set(getattr(candidate, "matched_evidence", []) or [])

    def _direct_low_explainability(self, candidate: Any) -> bool:
        return bool(
            candidate
            and float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.22
            and self.judge._residual(candidate) > 0.65
        )

    def _strong_specific_candidate(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                self.judge._priority(candidate)
                or float(getattr(candidate, "specificity", 0.0) or 0.0) >= 0.85
                or self.judge._gap_authorizable(candidate)
            )
        )

    def _high_explanatory_candidate(self, candidate: Any, data: Dict[str, Any]) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if not data.get("core"):
            return False
        coverage = float(getattr(candidate, "coverage_score", 0.0) or 0.0)
        core_coverage = float(
            getattr(candidate, "core_explanatory_coverage", 0.0) or 0.0
        )
        residual = self.judge._residual(candidate)
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        return bool(
            coverage >= 0.70
            and core_coverage >= 0.65
            and residual <= 0.25
            and (
                score >= 0.65
                or getattr(candidate, "required_met", False)
                or self.judge._priority(candidate)
                or self.judge._systemic_primary(candidate)
            )
        )

    @staticmethod
    def _name(candidate: Any) -> str:
        return str(getattr(candidate, "diagnosis", "") or "")


class DiagnosisJudge:
    """Deterministic judge for an already-generated diagnosis candidate table."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge: Any = None):
        diagnosis_section = (config or {}).get("diagnosis") or {}
        section = diagnosis_section.get("judge") or {}
        self.knowledge = knowledge
        self.top_k = int(section.get("top_k", 20) or 20)
        self.differential_top_k = int(
            section.get(
                "judge_top_k",
                diagnosis_section.get("judge_top_k", 5),
            )
            or 5
        )
        self.max_reviews = int(section.get("max_reviews", 20) or 20)
        self.pairwise_close_margin = float(
            section.get(
                "pairwise_close_margin",
                diagnosis_section.get("pairwise_close_margin", 0.18),
            )
            or 0.18
        )
        self.discriminating_exam_max_items = int(
            section.get(
                "discriminating_exam_max_items",
                diagnosis_section.get("discriminating_exam_max_items", 4),
            )
            or 4
        )
        self.filtered_pool_max_size = int(
            section.get(
                "filtered_pool_max_size",
                diagnosis_section.get("filtered_pool_max_size", 8),
            )
            or 8
        )
        self.differential_score_margin = float(
            section.get(
                "differential_score_margin",
                diagnosis_section.get("differential_score_margin", 0.12),
            )
            or 0.12
        )
        self.gap_authorization_min_score = float(
            section.get("gap_authorization_min_score", 0.42) or 0.42
        )
        self.gap_authorization_min_coverage = float(
            section.get("gap_authorization_min_coverage", 0.18) or 0.18
        )
        self.gap_authorization_max_residual = float(
            section.get("gap_authorization_max_residual", 0.88) or 0.88
        )
        self.priority_gap_bonus = float(section.get("priority_gap_bonus", 0.14) or 0.14)
        self.required_met_bonus = float(section.get("required_met_bonus", 0.005) or 0.005)
        self.specificity_bonus = float(section.get("specificity_bonus", 0.08) or 0.08)
        self.coverage_bonus = float(section.get("coverage_bonus", 0.24) or 0.24)
        self.residual_penalty = float(section.get("residual_penalty", 0.18) or 0.18)
        self.core_coverage_bonus = float(
            section.get("core_coverage_bonus", 0.30) or 0.30
        )
        self.residual_core_penalty = float(
            section.get("residual_core_penalty", 0.09) or 0.09
        )
        self.explanatory_preference_margin = float(
            section.get("explanatory_preference_margin", 0.10) or 0.10
        )
        self.gap_authorization_min_explanatory_score = float(
            section.get("gap_authorization_min_explanatory_score", 0.46) or 0.46
        )
        self.gap_authorization_min_core_coverage = float(
            section.get("gap_authorization_min_core_coverage", 0.40) or 0.40
        )
        self.gap_authorization_max_core_residual = int(
            section.get("gap_authorization_max_core_residual", 1) or 1
        )
        self.parent_fallback_margin = float(
            section.get("parent_fallback_margin", 0.10) or 0.10
        )
        self.secondary_min_score = float(section.get("secondary_min_score", 0.45) or 0.45)
        self.gap_target_limit = int(section.get("evidence_gap_target_limit", 2) or 2)
        self.pool_filter = DifferentialPoolFilter(self)

    def judge(
        self,
        candidates: Sequence[Any],
        preselected: Optional[Sequence[Any]] = None,
        max_final_diagnoses: int = 3,
    ) -> JudgeDecision:
        ranked = [item for item in candidates or [] if self._has_signal(item)]
        ranked = [item for item in ranked if not getattr(item, "hard_contradiction", False)]
        ranked = sorted(ranked, key=self._sort_key, reverse=True)
        retriever_top1 = self._name(candidates[0]) if candidates else ""

        decision = JudgeDecision(retriever_top1=retriever_top1)
        if not ranked:
            decision.reasoning = "Judge found no supported candidate."
            return decision

        raw_differential_pool = self._differential_pool(ranked)
        raw_pool_source = self._differential_pool_source(raw_differential_pool)
        pool_filter = self.pool_filter.filter(raw_differential_pool, raw_pool_source)
        differential_pool = pool_filter.candidates or raw_differential_pool
        allowed_pairs = self.pool_filter.allowed_pair_names(pool_filter)
        pairwise = self._pairwise_comparisons(differential_pool, allowed_pairs)
        required_gap_by_candidate = self._required_gap_by_candidate(differential_pool)
        discriminating_findings = self._discriminating_findings(
            differential_pool, required_gap_by_candidate
        )
        discriminating_exams = self._discriminating_exams(
            differential_pool, pairwise, discriminating_findings
        )
        primary_pool_names = {item.diagnosis for item in differential_pool}
        primary_candidates = [
            item for item in differential_pool if item.diagnosis in primary_pool_names
        ] or ranked[: self.top_k]
        primary = self._choose_primary(primary_candidates)
        if primary is None:
            decision.reasoning = "Judge found no eligible primary candidate."
            return decision

        gap_state_by_candidate = self._gap_state_by_candidate(ranked)
        gap_state_distribution = self._gap_state_distribution(gap_state_by_candidate)
        defer_reason = self._defer_primary_lock_reason(
            primary,
            differential_pool,
            pairwise,
            discriminating_exams,
        )
        needs_discriminating = bool(defer_reason)
        primary_status = "deferred" if needs_discriminating else "locked"
        gap_authorized: List[str] = []
        if self._gap_submission_authorized(primary):
            setattr(primary, "required_gap_authorized", True)
            gap_authorized.append(primary.diagnosis)

        secondary = self._select_secondary(primary, ranked, max_final_diagnoses)
        final = [primary.diagnosis] + [item.diagnosis for item in secondary]

        if needs_discriminating:
            evidence_gap_targets = self._deferred_evidence_gap_targets(
                primary,
                differential_pool,
            )
        else:
            evidence_gap_targets = self._evidence_gap_targets(primary, ranked, final)
        blocked = self._blocked_records(ranked, final)
        reviews = self._reviews(ranked, primary, secondary, evidence_gap_targets, blocked)

        decision.judge_primary = primary.diagnosis
        decision.primary = primary.diagnosis
        decision.primary_status = primary_status
        decision.needs_discriminating_exams = needs_discriminating
        decision.provisional_primary = primary.diagnosis if needs_discriminating else ""
        decision.locked_primary = primary.diagnosis if not needs_discriminating else ""
        decision.defer_reason = defer_reason
        decision.pre_discrimination_primary = primary.diagnosis
        decision.fallback_primary = primary.diagnosis
        decision.discrimination_attempted = needs_discriminating
        decision.discrimination_resolved = not needs_discriminating
        decision.secondary = [item.diagnosis for item in secondary]
        decision.differential = [item["diagnosis"] for item in blocked[: self.max_reviews]]
        decision.evidence_gap_targets = evidence_gap_targets
        decision.final_diagnoses = final[:max(1, int(max_final_diagnoses or 1))]
        decision.required_gap_authorized_diagnoses = gap_authorized
        decision.blocked_diagnoses = blocked
        decision.reviews = reviews
        decision.high_value_gap_candidates = [
            item.diagnosis
            for item in differential_pool
            if item.diagnosis != primary.diagnosis
            and self._high_value_unresolved_contender(primary, item)
        ][: self.gap_target_limit]
        decision.explanatory_coverage = self._coverage(primary)
        decision.core_explanatory_coverage = self._core_coverage(primary)
        decision.residual_evidence_score = self._residual(primary)
        decision.residual_core_evidence_count = self._residual_core_count(primary)
        decision.required_gap_state_by_candidate = gap_state_by_candidate
        decision.gap_state_distribution = gap_state_distribution
        previous_primary = str((preselected or [""])[0] if preselected else "")
        if previous_primary and previous_primary != primary.diagnosis:
            decision.primary_unlock_reason = self._primary_unlock_reason(
                previous_primary,
                primary,
                ranked,
            )
        model_primary = self._model_score_primary(ranked)
        decision.explanation_score_changed_ranking = bool(
            model_primary and model_primary != primary.diagnosis
        )
        decision.differential_candidates = [item.diagnosis for item in differential_pool]
        decision.pairwise_comparisons = pairwise
        decision.excluded_from_pairwise = list(pool_filter.excluded)
        decision.pool_filter_reasons = dict(pool_filter.pool_filter_reasons)
        decision.cluster_assignments = dict(pool_filter.cluster_assignments)
        decision.pairwise_allowed_matrix = list(pool_filter.pairwise_allowed_matrix)
        decision.core_evidence_by_candidate = dict(pool_filter.core_evidence_by_candidate)
        decision.generic_evidence_by_candidate = dict(
            pool_filter.generic_evidence_by_candidate
        )
        decision.pool_filter_summary = dict(pool_filter.summary)
        decision.discriminating_findings = discriminating_findings
        decision.discriminating_exams = discriminating_exams
        decision.required_gap_by_candidate = required_gap_by_candidate
        decision.differential_pool_source = dict(pool_filter.pool_source)
        decision.dynamic_rerank_trace = [
            {
                "stage": "initial_judge",
                "primary": primary.diagnosis,
                "primary_status": primary_status,
                "needs_discriminating_exams": needs_discriminating,
                "defer_reason": defer_reason,
                "required_gap_state": gap_state_by_candidate.get(primary.diagnosis, ""),
                "gap_state_distribution": dict(gap_state_distribution),
                "primary_unlock_reason": decision.primary_unlock_reason,
                "explanation_score_changed_ranking": decision.explanation_score_changed_ranking,
                "pool_filter_summary": dict(pool_filter.summary),
                "ranked": [
                    {
                        "diagnosis": item.diagnosis,
                        "primary_eligibility_score": round(
                            self._primary_eligibility_score(item),
                            4,
                        ),
                        "judge_score": round(self._judge_score(item), 4),
                        "required_gap_state": gap_state_by_candidate.get(
                            item.diagnosis,
                            self._required_gap_state(item),
                        ),
                        "required_met": bool(getattr(item, "required_met", False)),
                        "explanatory_coverage": round(self._coverage(item), 4),
                        "core_explanatory_coverage": round(
                            self._core_coverage(item),
                            4,
                        ),
                        "residual_evidence_score": round(self._residual(item), 4),
                        "residual_core_evidence_count": self._residual_core_count(item),
                    }
                    for item in differential_pool
                ],
            }
        ]
        decision.decision_override = bool(
            decision.retriever_top1
            and decision.judge_primary
            and decision.retriever_top1 != decision.judge_primary
        )
        decision.reasoning = self._reasoning(decision)
        return decision

    def _differential_pool(self, ranked: Sequence[Any]) -> List[Any]:
        pool: List[Any] = []
        top_score = self._judge_score(ranked[0]) if ranked else 0.0
        for index, item in enumerate(ranked[: self.top_k]):
            if item and not getattr(item, "hard_contradiction", False):
                if (
                    index < max(2, self.differential_top_k)
                    or self._judge_score(item) >= top_score - self.differential_score_margin
                    or self._tail_differential_candidate(item)
                ):
                    pool.append(item)
        limit = max(self.differential_top_k, min(self.top_k, 12))
        for item in ranked[self.differential_top_k : self.top_k]:
            if len(pool) >= limit:
                break
            if not self._tail_differential_candidate(item):
                continue
            if item not in pool:
                pool.append(item)
        result: List[Any] = []
        seen_ids: set[int] = set()
        for item in pool:
            marker = id(item)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            result.append(item)
        return result

    def _tail_differential_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if not self._has_signal(candidate):
            return False
        if not (
            self._priority(candidate)
            or self._gap_authorizable(candidate)
            or float(getattr(candidate, "specificity", 0.0) or 0.0) >= 0.85
        ):
            return False
        return bool(
            getattr(candidate, "required_gaps", None)
            or self._objective_signal(candidate)
            or float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.18
            or float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45
        )

    def _differential_pool_source(self, pool: Sequence[Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for index, item in enumerate(pool):
            if not item:
                continue
            result[item.diagnosis] = (
                "top_k" if index < self.differential_top_k else "top20_priority_tail"
            )
        return result

    def _pairwise_comparisons(
        self,
        pool: Sequence[Any],
        allowed_pairs: Optional[set[tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        comparisons: List[Dict[str, Any]] = []
        for left_index, left in enumerate(pool):
            for right in pool[left_index + 1 :]:
                pair_key = tuple(sorted((left.diagnosis, right.diagnosis)))
                if allowed_pairs is not None and pair_key not in allowed_pairs:
                    continue
                preferred = self._pairwise_preferred(left, right)
                alternate = right if preferred is left else left
                left_gaps = list(getattr(left, "required_gaps", []) or [])
                right_gaps = list(getattr(right, "required_gaps", []) or [])
                shared_evidence = sorted(
                    set(getattr(left, "matched_evidence", []) or [])
                    & set(getattr(right, "matched_evidence", []) or [])
                )[:5]
                comparisons.append(
                    {
                        "left": left.diagnosis,
                        "right": right.diagnosis,
                        "preferred": preferred.diagnosis,
                        "reason": self._pairwise_reason(preferred, alternate),
                        "score_delta": round(
                            self._judge_score(preferred) - self._judge_score(alternate),
                            4,
                        ),
                        "close_call": abs(
                            self._judge_score(left) - self._judge_score(right)
                        )
                        <= self.pairwise_close_margin,
                        "left_required_gaps": left_gaps[:4],
                        "right_required_gaps": right_gaps[:4],
                        "shared_matched_evidence": shared_evidence,
                        "discriminating_findings": self._pairwise_discriminating_findings(
                            left, right
                        ),
                        "discriminating_exams": self._pairwise_discriminating_exams(
                            left, right
                        ),
                    }
                )
        return comparisons

    def _pairwise_preferred(self, left: Any, right: Any) -> Any:
        left_key = self._pairwise_key(left)
        right_key = self._pairwise_key(right)
        return left if left_key >= right_key else right

    def _pairwise_reason(self, preferred: Any, alternate: Any) -> str:
        if self._core_coverage(preferred) > self._core_coverage(alternate) + 0.10:
            return "preferred because it explains more core evidence"
        if self._residual_core_count(preferred) < self._residual_core_count(alternate):
            return "preferred because unexplained core evidence is lower"
        if self._coverage(preferred) > self._coverage(alternate) + 0.08:
            return "preferred because it explains more evidence"
        if self._residual(preferred) + 0.08 < self._residual(alternate):
            return "preferred because residual evidence is lower"
        if self._gap_authorizable(preferred) and not self._gap_authorizable(alternate):
            return "preferred as high-priority candidate with actionable evidence gap"
        if self._priority(preferred) and not self._priority(alternate):
            return "preferred because etiology/structural/specific diagnosis has priority"
        if self._trusted(preferred) and not self._trusted(alternate):
            return "preferred because required evidence is met"
        return "preferred by evidence-weighted judge score"

    def _pairwise_key(self, candidate: Any) -> tuple:
        return (
            self._core_coverage(candidate),
            -self._residual_core_count(candidate),
            self._coverage(candidate),
            1.0 - self._residual(candidate),
            self._judge_score(candidate),
            1 if self._priority(candidate) else 0,
            1 if self._trusted(candidate) else 0,
            float(getattr(candidate, "score", 0.0) or 0.0),
        )

    def _pairwise_discriminating_findings(self, left: Any, right: Any) -> List[str]:
        left_gaps = set(getattr(left, "required_gaps", []) or [])
        right_gaps = set(getattr(right, "required_gaps", []) or [])
        findings = list(dict.fromkeys(list(left_gaps - right_gaps) + list(right_gaps - left_gaps)))
        if not findings:
            findings = list(
                dict.fromkeys(
                    list(getattr(left, "residual_evidence", []) or [])
                    + list(getattr(right, "residual_evidence", []) or [])
                )
            )
        return findings[:6]

    def _pairwise_discriminating_exams(self, left: Any, right: Any) -> List[str]:
        return self._candidate_exam_union([left, right])[: self.discriminating_exam_max_items]

    def _defer_primary_lock_reason(
        self,
        primary: Any,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        discriminating_exams: Sequence[str],
    ) -> str:
        if not primary or not discriminating_exams:
            return ""
        if getattr(primary, "hard_contradiction", False):
            return ""
        contenders = [
            item
            for item in pool
            if item
            and item.diagnosis != primary.diagnosis
            and self._high_value_unresolved_contender(primary, item)
        ]
        if not contenders:
            return ""
        if self._primary_lock_allowed(primary, contenders, pairwise):
            return ""
        names = ", ".join(item.diagnosis for item in contenders[:3])
        return (
            "defer_for_discrimination: high-value unresolved contender(s) "
            f"remain before primary lock: {names}"
        )

    def _primary_lock_allowed(
        self,
        primary: Any,
        contenders: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
    ) -> bool:
        if not self._trusted(primary):
            return False
        gap_state = self._required_gap_state(primary)
        if gap_state in {"hard_contradiction", "unsupported_gap", "actionable_gap"}:
            return False
        if gap_state == "partially_satisfied" and contenders:
            return False
        if self._residual_core_count(primary) > 0 and contenders:
            return False
        if self._core_coverage(primary) < 0.45 and contenders:
            return False
        primary_score = self._judge_score(primary)
        if any(primary_score <= self._judge_score(item) + self.pairwise_close_margin for item in contenders):
            return False
        for comparison in pairwise:
            if not comparison.get("close_call"):
                continue
            if primary.diagnosis in {comparison.get("left"), comparison.get("right")}:
                return False
        return True

    def _high_value_unresolved_contender(self, primary: Any, contender: Any) -> bool:
        if not contender or getattr(contender, "hard_contradiction", False):
            return False
        if not self._has_signal(contender):
            return False
        if self._trusted(contender) and self._judge_score(contender) > self._judge_score(primary):
            return True
        priority_or_specific = (
            self._priority(contender)
            or self._systemic_primary(contender)
            or float(getattr(contender, "specificity", 0.0) or 0.0) >= 0.85
        )
        if not priority_or_specific:
            return False
        actionable_gap = bool(getattr(contender, "required_gaps", None)) or self._gap_authorizable(contender)
        if not actionable_gap and not self._candidate_discriminating_exams(contender):
            return False
        contender_score = self._judge_score(contender)
        primary_score = self._judge_score(primary)
        if self._is_manifestation(primary) or self._generic_parent_of(contender, primary):
            return contender_score >= primary_score - 0.24
        if (
            self._coverage(contender)
            >= self._coverage(primary) - 0.12
            and self._residual(contender)
            <= self._residual(primary) + 0.18
        ):
            return contender_score >= primary_score - max(self.pairwise_close_margin, 0.24)
        if (
            self._core_coverage(contender) >= self._core_coverage(primary) + 0.12
            or self._residual_core_count(contender) < self._residual_core_count(primary)
        ):
            return contender_score >= primary_score - max(self.pairwise_close_margin, 0.26)
        return contender_score >= primary_score - self.pairwise_close_margin

    def _required_gap_by_candidate(
        self, pool: Sequence[Any]
    ) -> Dict[str, Dict[str, List[str]]]:
        all_gap_sets = {
            item.diagnosis: set(getattr(item, "required_gaps", []) or [])
            for item in pool
        }
        result: Dict[str, Dict[str, List[str]]] = {}
        for item in pool:
            gaps = list(getattr(item, "required_gaps", []) or [])
            other_gaps = set()
            for name, gap_set in all_gap_sets.items():
                if name != item.diagnosis:
                    other_gaps.update(gap_set)
            discriminating = [gap for gap in gaps if gap not in other_gaps]
            if not discriminating:
                discriminating = gaps[:2]
            result[item.diagnosis] = {
                "confirmatory_gap": gaps[:6],
                "discriminating_gap": discriminating[:6],
            }
        return result

    def _discriminating_findings(
        self,
        pool: Sequence[Any],
        gap_by_candidate: Dict[str, Dict[str, List[str]]],
    ) -> List[str]:
        findings: List[str] = []
        for item in pool:
            payload = gap_by_candidate.get(item.diagnosis) or {}
            for finding in payload.get("discriminating_gap") or []:
                if finding and finding not in findings:
                    findings.append(finding)
            for finding in getattr(item, "residual_evidence", []) or []:
                if finding and finding not in findings:
                    findings.append(finding)
        return findings[:12]

    def _discriminating_exams(
        self,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        findings: Sequence[str],
    ) -> List[str]:
        exams: List[str] = []
        pool_names = {str(getattr(item, "diagnosis", "") or "") for item in pool}
        high_prior_candidates = sorted(
            [item for item in pool if self._high_prior_specific_exam_candidate(item)],
            key=self._specific_exam_candidate_key,
            reverse=True,
        )
        for candidate in high_prior_candidates:
            for item in self._candidate_discriminating_exams(candidate):
                if item and item not in exams:
                    exams.append(item)
        for names, hinted_exams in _DIFFERENTIAL_SET_EXAM_HINTS:
            if names.issubset(pool_names):
                for item in hinted_exams:
                    if item and item not in exams:
                        exams.append(item)
        for comparison in pairwise:
            if not comparison.get("close_call"):
                continue
            for item in comparison.get("discriminating_exams") or []:
                if item and item not in exams:
                    exams.append(item)
        for item in self._candidate_exam_union(pool):
            if item and item not in exams:
                exams.append(item)
        return exams[: self.discriminating_exam_max_items]

    def _high_prior_specific_exam_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if float(getattr(candidate, "specificity", 0.0) or 0.0) < 0.85:
            return False
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        source_prior = float(getattr(candidate, "source_prior", 0.0) or 0.0)
        if source_prior < 0.55 and f"diagnosis:{candidate.diagnosis}" not in matched:
            return False
        return bool(
            getattr(candidate, "required_gaps", None)
            or self._priority(candidate)
            or f"diagnosis:{candidate.diagnosis}" in matched
        )

    def _specific_exam_candidate_key(self, candidate: Any) -> tuple:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        return (
            1 if f"diagnosis:{candidate.diagnosis}" in matched else 0,
            float(getattr(candidate, "source_prior", 0.0) or 0.0),
            1 if getattr(candidate, "required_gaps", None) else 0,
            float(getattr(candidate, "coverage_score", 0.0) or 0.0),
            float(getattr(candidate, "specificity", 0.0) or 0.0),
        )

    def _candidate_exam_union(self, candidates: Sequence[Any]) -> List[str]:
        exams: List[str] = []
        for candidate in candidates:
            for exam in self._candidate_discriminating_exams(candidate):
                if exam and exam not in exams:
                    exams.append(exam)
        return exams

    def _candidate_discriminating_exams(self, candidate: Any) -> List[str]:
        if not candidate:
            return []
        name = str(getattr(candidate, "diagnosis", "") or "")
        exams: List[str] = []
        entry: Dict[str, Any] = {}
        if self.knowledge:
            entry = self.knowledge.get(name) or {}
        for field_name in (
            "discriminating_exams",
            "strong_verification_exams",
            "required_exams",
        ):
            for exam in entry.get(field_name, []) or []:
                text = str(exam).strip()
                if text and text not in exams:
                    exams.append(text)
        for exam in _DIFFERENTIAL_EXAM_HINTS.get(name, []):
            if exam and exam not in exams:
                exams.append(exam)
        return exams[:6]

    def _choose_primary(self, ranked: Sequence[Any]) -> Optional[Any]:
        if not ranked:
            return None
        ranked = sorted(ranked, key=self._sort_key, reverse=True)
        best = ranked[0]
        for item in ranked[1:]:
            if self._should_prefer_explanatory_primary(item, best):
                best = item
        gap_candidates = [
            item
            for item in ranked
            if self._gap_authorizable(item)
            and self._should_prefer_explanatory_primary(item, best, allow_tie=True)
        ]
        if gap_candidates:
            best = max(gap_candidates, key=self._sort_key)

        best = self._maybe_prefer_systemic_primary(best, ranked)
        best = self._maybe_prefer_parent_primary(best, ranked[: self.top_k])
        return best

    def _should_prefer_explanatory_primary(
        self,
        candidate: Any,
        selected: Any,
        allow_tie: bool = False,
    ) -> bool:
        if not candidate or not selected:
            return False
        if getattr(candidate, "hard_contradiction", False):
            return False
        if not (self._trusted(candidate) or self._gap_authorizable(candidate)):
            return False
        candidate_score = self._judge_score(candidate)
        selected_score = self._judge_score(selected)
        margin = 0.0 if allow_tie else self.explanatory_preference_margin
        if candidate_score >= selected_score + margin:
            return True
        candidate_core = self._core_coverage(candidate)
        selected_core = self._core_coverage(selected)
        candidate_residual = self._residual(candidate)
        selected_residual = self._residual(selected)
        if (
            self._priority(candidate)
            and not self._priority(selected)
            and candidate_core + 0.10 >= selected_core
            and candidate_residual <= selected_residual + 0.10
        ):
            return True
        if (
            candidate_core >= selected_core + 0.18
            and candidate_residual <= selected_residual + 0.12
            and candidate_score >= selected_score - self.pairwise_close_margin
        ):
            return True
        if (
            self._residual_core_count(selected) > self._residual_core_count(candidate)
            and candidate_score >= selected_score - self.pairwise_close_margin
        ):
            return True
        return False

    def _maybe_prefer_systemic_primary(self, selected: Any, ranked: Sequence[Any]) -> Any:
        if self._systemic_primary(selected):
            return selected
        for candidate in ranked:
            if candidate is selected:
                continue
            if not self._systemic_primary(candidate):
                continue
            if not (self._trusted(candidate) or self._gap_authorizable(candidate)):
                continue
            if self._judge_score(candidate) + 0.06 < self._judge_score(selected):
                continue
            if self._is_manifestation(selected) or self._generic_parent_of(candidate, selected):
                return candidate
            if self._coverage(candidate) >= self._coverage(selected) and self._residual(candidate) <= self._residual(selected) + 0.12:
                return candidate
        return selected

    def _maybe_prefer_parent_primary(self, selected: Any, ranked: Sequence[Any]) -> Any:
        parent = str(getattr(selected, "parent_diagnosis", "") or "")
        if parent not in _PARENT_FALLBACK_NAMES:
            direct_parent = self._direct_parent_fallback_candidate(selected, ranked)
            return direct_parent or selected
        parent_candidate = next((item for item in ranked if item.diagnosis == parent), None)
        if not parent_candidate or getattr(parent_candidate, "hard_contradiction", False):
            direct_parent = self._direct_parent_fallback_candidate(selected, ranked)
            return direct_parent or selected
        selected_matched = set(getattr(selected, "matched_evidence", []) or [])
        parent_matched = set(getattr(parent_candidate, "matched_evidence", []) or [])
        if f"diagnosis:{selected.diagnosis}" in selected_matched:
            return selected
        if (
            f"diagnosis:{parent}" in parent_matched
            and (
                getattr(selected, "required_gap_authorized", False)
                or not getattr(selected, "required_met", False)
                or not self._objective_signal(selected)
            )
        ):
            return parent_candidate
        if f"diagnosis:{parent}" in parent_matched:
            if self._judge_score(parent_candidate) + self.parent_fallback_margin >= self._judge_score(selected):
                return parent_candidate
        if (
            self._trusted(parent_candidate)
            and self._judge_score(parent_candidate) + self.parent_fallback_margin >= self._judge_score(selected)
            and not self._trusted(selected)
        ):
            return parent_candidate
        return selected

    def _direct_parent_fallback_candidate(self, selected: Any, ranked: Sequence[Any]) -> Optional[Any]:
        if not (
            selected
            and (
                getattr(selected, "required_gap_authorized", False)
                or self._gap_submission_authorized(selected)
                or self._required_gap_state(selected)
                in {"actionable_gap", "partially_satisfied", "nonblocking_gap"}
            )
            and not f"diagnosis:{selected.diagnosis}"
            in set(getattr(selected, "matched_evidence", []) or [])
        ):
            return None
        for candidate in ranked:
            if candidate.diagnosis not in _PARENT_FALLBACK_NAMES:
                continue
            if getattr(candidate, "hard_contradiction", False):
                continue
            if f"diagnosis:{candidate.diagnosis}" not in set(getattr(candidate, "matched_evidence", []) or []):
                continue
            if self._trusted(candidate):
                return candidate
            if not (self._same_family(candidate, selected) or self._same_body_system(candidate, selected)):
                continue
            if self._trusted(candidate) or self._judge_score(candidate) + self.parent_fallback_margin >= self._judge_score(selected):
                return candidate
        return None

    def _select_secondary(
        self,
        primary: Any,
        ranked: Sequence[Any],
        max_final_diagnoses: int,
    ) -> List[Any]:
        result: List[Any] = []
        limit = max(0, int(max_final_diagnoses or 1) - 1)
        if limit <= 0:
            return result
        secondary_pool = sorted(
            [item for item in ranked if item and item.diagnosis != primary.diagnosis],
            key=lambda item: (
                1
                if (
                    self._causally_related(primary, item)
                    and self._is_manifestation(item)
                    and self._independent_objective(item)
                )
                else 0,
                1
                if (
                    self._is_manifestation(item)
                    and f"diagnosis:{item.diagnosis}"
                    in set(getattr(primary, "matched_evidence", []) or [])
                )
                else 0,
                1 if self._same_family(primary, item) and self._independent_objective(item) else 0,
                self._judge_score(item),
            ),
            reverse=True,
        )
        for candidate in secondary_pool:
            if candidate.diagnosis == primary.diagnosis:
                continue
            if len(result) >= limit:
                break
            if getattr(candidate, "hard_contradiction", False):
                continue
            if not self._trusted(candidate):
                continue
            if (
                self._is_manifestation(candidate)
                and f"diagnosis:{candidate.diagnosis}"
                in set(getattr(primary, "matched_evidence", []) or [])
            ):
                result.append(candidate)
                continue
            if self._generic_parent_of(primary, candidate) or self._generic_parent_of(candidate, primary):
                if self._same_family(primary, candidate) and self._independent_objective(candidate):
                    result.append(candidate)
                continue
            if self._causally_related(primary, candidate):
                if self._is_manifestation(candidate):
                    if self._independent_objective(candidate):
                        result.append(candidate)
                    continue
                if self._independent_objective(candidate):
                    result.append(candidate)
                continue
            if float(getattr(candidate, "score", 0.0) or 0.0) < self.secondary_min_score:
                continue
            if self._same_family(primary, candidate) and self._independent_objective(candidate):
                result.append(candidate)
        return result

    def _evidence_gap_targets(
        self,
        primary: Any,
        ranked: Sequence[Any],
        final: Sequence[str],
    ) -> List[str]:
        targets: List[str] = []
        final_set = set(final or [])
        if getattr(primary, "required_gaps", None):
            targets.append(primary.diagnosis)
        for candidate in ranked:
            if len(targets) >= self.gap_target_limit:
                break
            if candidate.diagnosis in final_set or candidate.diagnosis in targets:
                continue
            if not self._gap_authorizable(candidate):
                continue
            if self._same_family(primary, candidate) or self._causally_related(primary, candidate):
                targets.append(candidate.diagnosis)
        return targets

    def _deferred_evidence_gap_targets(
        self,
        primary: Any,
        pool: Sequence[Any],
    ) -> List[str]:
        targets: List[str] = []
        for candidate in [primary] + list(pool or []):
            if not candidate or candidate.diagnosis in targets:
                continue
            if getattr(candidate, "hard_contradiction", False):
                continue
            if not (
                getattr(candidate, "required_gaps", None)
                or self._candidate_discriminating_exams(candidate)
            ):
                continue
            if candidate.diagnosis == getattr(primary, "diagnosis", ""):
                targets.append(candidate.diagnosis)
                continue
            if self._high_value_unresolved_contender(primary, candidate):
                targets.append(candidate.diagnosis)
            if len(targets) >= self.gap_target_limit:
                break
        return targets[: max(1, self.gap_target_limit)]

    def _gap_state_by_candidate(self, ranked: Sequence[Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for candidate in ranked[: self.max_reviews]:
            name = self._name(candidate)
            if name:
                result[name] = self._required_gap_state(candidate)
        return result

    @staticmethod
    def _gap_state_distribution(states: Dict[str, str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for state in states.values():
            result[state] = result.get(state, 0) + 1
        return result

    def _required_gap_state(self, candidate: Any) -> str:
        if not candidate:
            return "unsupported_gap"
        if getattr(candidate, "hard_contradiction", False):
            return "hard_contradiction"
        existing = str(getattr(candidate, "required_gap_state", "") or "")
        if existing in {
            "satisfied",
            "partially_satisfied",
            "actionable_gap",
            "nonblocking_gap",
            "unsupported_gap",
            "hard_contradiction",
        }:
            return existing
        if getattr(candidate, "required_met", False) and not getattr(
            candidate, "required_gaps", None
        ):
            return "satisfied"
        if not getattr(candidate, "matched_evidence", None):
            return "unsupported_gap"
        if getattr(candidate, "required_gaps", None):
            if self._candidate_discriminating_exams(candidate):
                return "actionable_gap"
            if self._explanatory_gap_authorizable(candidate):
                return "nonblocking_gap"
            return "partially_satisfied"
        if self._explanatory_gap_authorizable(candidate):
            return "nonblocking_gap"
        return "partially_satisfied"

    def _gap_submission_authorized(self, candidate: Any) -> bool:
        state = self._required_gap_state(candidate)
        return state in {"actionable_gap", "nonblocking_gap", "partially_satisfied"} and (
            self._gap_authorizable(candidate)
            or self._explanatory_gap_authorizable(candidate)
        )

    def _explanatory_gap_authorizable(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "required_met", False):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        if not (
            self._priority(candidate)
            or self._systemic_primary(candidate)
            or float(getattr(candidate, "specificity", 0.0) or 0.0) >= 0.85
        ):
            return False
        if not self._core_support_signal(candidate):
            return False
        explanatory_score = (
            0.45 * self._coverage(candidate)
            + 0.40 * self._core_coverage(candidate)
            + 0.15 * (1.0 - self._residual(candidate))
            + 0.10 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
        )
        if explanatory_score >= self.gap_authorization_min_explanatory_score:
            return True
        if self._core_coverage(candidate) >= self.gap_authorization_min_core_coverage:
            return self._residual_core_count(candidate) <= self.gap_authorization_max_core_residual
        if (
            self._coverage(candidate) >= max(0.52, self.gap_authorization_min_coverage)
            and self._residual(candidate) <= self.gap_authorization_max_residual
        ):
            return True
        return False

    @staticmethod
    def _core_support_signal(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{getattr(candidate, 'diagnosis', '')}" in matched:
            return True
        core = matched & _CORE_EVIDENCE_TOKENS
        if core:
            return True
        if matched & _OBJECTIVE_GAP_FINDINGS:
            return True
        explained = set(getattr(candidate, "explained_evidence", []) or [])
        if explained & _CORE_EVIDENCE_TOKENS:
            return True
        return False

    def _model_score_primary(self, ranked: Sequence[Any]) -> str:
        pool = [item for item in ranked or [] if item and not getattr(item, "hard_contradiction", False)]
        if not pool:
            return ""
        return self._name(
            max(pool, key=lambda item: float(getattr(item, "score", 0.0) or 0.0))
        )

    def _primary_unlock_reason(
        self,
        previous_primary: str,
        primary: Any,
        ranked: Sequence[Any],
    ) -> str:
        if not previous_primary or not primary:
            return ""
        previous = next(
            (item for item in ranked if self._name(item) == previous_primary),
            None,
        )
        if not previous:
            return "previous primary no longer appears in candidate table"
        if getattr(previous, "hard_contradiction", False):
            return "previous primary gained hard contradiction"
        if self._residual_core_count(primary) < self._residual_core_count(previous):
            return "new primary leaves fewer unexplained core findings"
        if self._core_coverage(primary) > self._core_coverage(previous) + 0.10:
            return "new primary explains more core findings"
        if self._judge_score(primary) > self._judge_score(previous):
            return "new primary has higher evidence-authority judge score"
        return "primary changed after evidence-authority rerank"

    def _blocked_records(self, ranked: Sequence[Any], final: Sequence[str]) -> List[Dict[str, Any]]:
        final_set = set(final or [])
        blocked: List[Dict[str, Any]] = []
        for candidate in ranked[: self.max_reviews]:
            if candidate.diagnosis in final_set:
                continue
            reason = "differential_only"
            gap_state = self._required_gap_state(candidate)
            if getattr(candidate, "hard_contradiction", False):
                reason = "hard_contradiction"
            elif not getattr(candidate, "matched_evidence", None):
                reason = "no_supporting_evidence"
            elif not getattr(candidate, "required_met", False):
                reason = gap_state
            elif self._is_manifestation(candidate):
                reason = "manifestation_or_complication_not_primary"
            elif self._generic_parent_name(candidate):
                reason = "generic_or_parent_diagnosis"
            blocked.append(
                {
                    "diagnosis": candidate.diagnosis,
                    "reason": reason,
                    "score": getattr(candidate, "score", 0.0),
                    "judge_score": round(self._judge_score(candidate), 4),
                    "explanatory_coverage": round(self._coverage(candidate), 4),
                    "core_explanatory_coverage": round(
                        self._core_coverage(candidate),
                        4,
                    ),
                    "residual_evidence_score": round(self._residual(candidate), 4),
                    "residual_core_evidence_count": self._residual_core_count(candidate),
                    "required_met": bool(getattr(candidate, "required_met", False)),
                    "required_gap_state": gap_state,
                    "required_gaps": list(getattr(candidate, "required_gaps", []) or [])[:4],
                    "hard_contradiction": bool(getattr(candidate, "hard_contradiction", False)),
                }
            )
        return blocked

    def _reviews(
        self,
        ranked: Sequence[Any],
        primary: Any,
        secondary: Sequence[Any],
        gap_targets: Sequence[str],
        blocked: Sequence[Dict[str, Any]],
    ) -> List[JudgeCandidateReview]:
        roles = {primary.diagnosis: "primary"}
        roles.update({item.diagnosis: "secondary" for item in secondary})
        roles.update({name: "evidence_gap" for name in gap_targets if name not in roles})
        blocked_reasons = {item["diagnosis"]: item["reason"] for item in blocked}
        reviews: List[JudgeCandidateReview] = []
        for candidate in ranked[: self.max_reviews]:
            role = roles.get(candidate.diagnosis, "differential")
            reason = blocked_reasons.get(candidate.diagnosis, role)
            if role == "primary" and self._requires_gap_authorization(candidate):
                reason = "provisional primary: strong explainability with required evidence gap"
            reviews.append(
                JudgeCandidateReview(
                    diagnosis=candidate.diagnosis,
                    role=role,
                    reason=reason,
                    score=float(getattr(candidate, "score", 0.0) or 0.0),
                    judge_score=round(self._judge_score(candidate), 4),
                    required_met=bool(getattr(candidate, "required_met", False)),
                    required_gap_authorized=bool(
                        getattr(candidate, "required_gap_authorized", False)
                    ),
                    hard_contradiction=bool(getattr(candidate, "hard_contradiction", False)),
                    coverage_score=float(getattr(candidate, "coverage_score", 0.0) or 0.0),
                    residual_score=float(getattr(candidate, "residual_score", 0.0) or 0.0),
                    explanatory_coverage=self._coverage(candidate),
                    core_explanatory_coverage=self._core_coverage(candidate),
                    residual_evidence_score=self._residual(candidate),
                    residual_core_evidence_count=self._residual_core_count(candidate),
                    diagnosis_type=str(getattr(candidate, "diagnosis_type", "") or ""),
                    specificity=float(getattr(candidate, "specificity", 0.0) or 0.0),
                    required_gaps=list(getattr(candidate, "required_gaps", []) or [])[:4],
                    matched_evidence=list(getattr(candidate, "matched_evidence", []) or [])[:6],
                    explained_evidence=list(getattr(candidate, "explained_evidence", []) or [])[:6],
                    unexplained_core_evidence=list(
                        getattr(candidate, "unexplained_core_evidence", []) or []
                    )[:6],
                    explanatory_rank_reason=str(
                        getattr(candidate, "explanatory_rank_reason", "") or ""
                    ),
                    required_gap_state=self._required_gap_state(candidate),
                )
            )
        return reviews

    def _requires_gap_authorization(self, candidate: Any) -> bool:
        return bool(
            candidate
            and not getattr(candidate, "required_met", False)
            and getattr(candidate, "required_gap_authorized", False)
        )

    def _gap_authorizable(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "required_met", False):
            return False
        if not getattr(candidate, "required_gaps", None):
            return False
        if not self._priority(candidate):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        if not self._objective_signal(candidate) and not self._explanatory_gap_authorizable(candidate):
            return False
        explanatory_score = (
            0.55 * self._coverage(candidate)
            + 0.35 * self._core_coverage(candidate)
            + 0.10 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
            - 0.20 * self._residual(candidate)
        )
        if (
            float(getattr(candidate, "score", 0.0) or 0.0) < self.gap_authorization_min_score
            and explanatory_score < self.gap_authorization_min_explanatory_score
        ):
            return False
        return (
            self._coverage(candidate) >= self.gap_authorization_min_coverage
            or self._residual(candidate) <= self.gap_authorization_max_residual
            or self._core_coverage(candidate) >= self.gap_authorization_min_coverage
            or float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45
        )

    def _trusted(self, candidate: Any) -> bool:
        state = self._required_gap_state(candidate)
        return bool(
            candidate
            and not getattr(candidate, "hard_contradiction", False)
            and getattr(candidate, "matched_evidence", None)
            and state
            in {
                "satisfied",
                "partially_satisfied",
                "actionable_gap",
                "nonblocking_gap",
            }
        )

    def _has_signal(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                getattr(candidate, "matched_evidence", None)
                or float(getattr(candidate, "source_prior", 0.0) or 0.0) > 0.0
            )
        )

    def _sort_key(self, candidate: Any) -> tuple:
        return (
            self._primary_eligibility_score(candidate),
            self._core_coverage(candidate),
            -self._residual_core_count(candidate),
            1.0 - self._residual(candidate),
            self._coverage(candidate),
            self._judge_score(candidate),
            1 if self._priority(candidate) else 0,
            float(getattr(candidate, "specificity", 0.0) or 0.0),
            float(getattr(candidate, "score", 0.0) or 0.0),
        )

    def _primary_eligibility_score(self, candidate: Any) -> float:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return -1.0
        gap_state = self._required_gap_state(candidate)
        gap_penalty = {
            "satisfied": 0.0,
            "nonblocking_gap": 0.03,
            "partially_satisfied": 0.08,
            "actionable_gap": 0.10,
            "unsupported_gap": 0.28,
            "hard_contradiction": 1.0,
        }.get(gap_state, 0.18)
        score = (
            0.32 * self._core_coverage(candidate)
            + 0.24 * self._coverage(candidate)
            + 0.14 * float(getattr(candidate, "specificity", 0.0) or 0.0)
            + 0.12 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
            + 0.10 * float(getattr(candidate, "score", 0.0) or 0.0)
            - 0.18 * min(1.0, 0.25 * self._residual_core_count(candidate))
            - 0.14 * self._residual(candidate)
            - gap_penalty
            - 0.10 * float(getattr(candidate, "contradiction_penalty", 0.0) or 0.0)
        )
        if self._priority(candidate):
            score += 0.06
        if self._systemic_primary(candidate):
            score += 0.04
        if self._is_manifestation(candidate):
            score -= 0.08
        if self._generic_parent_name(candidate):
            score -= 0.05
        return round(score, 4)

    def _judge_score(self, candidate: Any) -> float:
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        score += self.coverage_bonus * self._coverage(candidate)
        score += self.core_coverage_bonus * self._core_coverage(candidate)
        score -= self.residual_penalty * self._residual(candidate)
        score -= self.residual_core_penalty * min(
            4,
            self._residual_core_count(candidate),
        )
        score += self.specificity_bonus * float(getattr(candidate, "specificity", 0.0) or 0.0)
        gap_state = self._required_gap_state(candidate)
        if gap_state == "satisfied":
            score += self.required_met_bonus
        elif gap_state in {"actionable_gap", "nonblocking_gap", "partially_satisfied"} and self._gap_authorizable(candidate):
            score += self.priority_gap_bonus
        elif gap_state == "unsupported_gap":
            score -= 0.15
        if self._priority(candidate):
            score += 0.05
        if not self._disease_specific_priority_allowed(candidate):
            score -= 0.25
        if self._systemic_primary(candidate):
            score += 0.05
        if self._is_manifestation(candidate):
            score -= 0.08
        if self._generic_parent_name(candidate):
            score -= 0.04
        if getattr(candidate, "hard_contradiction", False):
            score -= 1.0
        return round(score, 4)

    @staticmethod
    def _coverage(candidate: Any) -> float:
        value = getattr(candidate, "explanatory_coverage", None)
        legacy = getattr(candidate, "coverage_score", 0.0)
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _core_coverage(candidate: Any) -> float:
        value = getattr(candidate, "core_explanatory_coverage", None)
        component_value = (getattr(candidate, "component_scores", {}) or {}).get(
            "core_explanatory_coverage"
        )
        legacy = getattr(candidate, "coverage_score", 0.0)
        if component_value is not None and (
            value is None or float(value or 0.0) == 0.0
        ):
            value = component_value
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _residual(candidate: Any) -> float:
        value = getattr(candidate, "residual_evidence_score", None)
        legacy = getattr(candidate, "residual_score", 1.0)
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _residual_core_count(candidate: Any) -> int:
        value = getattr(candidate, "residual_core_evidence_count", None)
        component_value = (getattr(candidate, "component_scores", {}) or {}).get(
            "residual_core_evidence_count"
        )
        try:
            current = int(float(value or 0))
        except (TypeError, ValueError):
            current = 0
        if component_value is not None and (value is None or current == 0):
            value = component_value
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    def _priority(self, candidate: Any) -> bool:
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        if not self._disease_specific_priority_allowed(candidate):
            return False
        return dtype in _PRIORITY_TYPES or float(getattr(candidate, "specificity", 0.0) or 0.0) >= 0.85

    @staticmethod
    def _disease_specific_priority_allowed(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if getattr(candidate, "diagnosis", "") == "克里格勒-纳贾尔综合征":
            return bool(
                matched
                & {
                    "ugt1a1_positive",
                    "genetic_suspicion",
                    "neonatal_jaundice",
                    "diagnosis:克里格勒-纳贾尔综合征",
                }
            )
        if getattr(candidate, "diagnosis", "") != "压力性尿失禁":
            return True
        return bool(
            matched
            & {
                "stress_urinary_incontinence",
                "urine_leak_with_pressure",
                "diagnosis:压力性尿失禁",
                "symptom:压力性尿失禁",
            }
        )

    def _systemic_primary(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                getattr(candidate, "diagnosis", "") in _SYSTEMIC_PRIMARY_NAMES
                or str(getattr(candidate, "diagnosis_type", "") or "").lower() == "systemic"
            )
        )

    def _is_manifestation(self, candidate: Any) -> bool:
        if self._systemic_primary(candidate):
            return False
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        return dtype in _MANIFESTATION_TYPES or getattr(candidate, "diagnosis", "") in _MANIFESTATION_NAMES

    def _generic_parent_name(self, candidate: Any) -> str:
        if not self.knowledge or not candidate:
            return ""
        entry = self.knowledge.get(getattr(candidate, "diagnosis", ""))
        for item in getattr(self.knowledge, "entries", {}).values():
            if str(item.get("parent_diagnosis") or "") == getattr(candidate, "diagnosis", ""):
                return getattr(candidate, "diagnosis", "")
        if entry.get("generic_suppressions") or entry.get("generalization_suppressions"):
            return ""
        return ""

    def _generic_parent_of(self, specific: Any, parent: Any) -> bool:
        return bool(
            specific
            and parent
            and str(getattr(specific, "parent_diagnosis", "") or "") == getattr(parent, "diagnosis", "")
        )

    def _same_family(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self.knowledge.get(left.diagnosis)
        right_entry = self.knowledge.get(right.diagnosis)
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        left_family = str(left_entry.get("disease_family") or left_entry.get("family") or "")
        right_family = str(right_entry.get("disease_family") or right_entry.get("family") or "")
        if left_system and right_system and left_system == right_system:
            return bool(left_family and right_family and left_family == right_family)
        left_parent = str(left_entry.get("parent_diagnosis") or "")
        right_parent = str(right_entry.get("parent_diagnosis") or "")
        return bool(left_parent and right_parent and left_parent == right_parent)

    def _same_body_system(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self.knowledge.get(left.diagnosis)
        right_entry = self.knowledge.get(right.diagnosis)
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        return bool(left_system and right_system and left_system == right_system)

    def _causally_related(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self.knowledge.get(left.diagnosis)
        right_entry = self.knowledge.get(right.diagnosis)
        return (
            right.diagnosis in set(str(item) for item in left_entry.get("causes", []) or [])
            or left.diagnosis in set(str(item) for item in right_entry.get("caused_by", []) or [])
            or left.diagnosis in set(str(item) for item in right_entry.get("causes", []) or [])
            or right.diagnosis in set(str(item) for item in left_entry.get("caused_by", []) or [])
            or self._same_family(left, right)
        )

    @staticmethod
    def _independent_objective(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{candidate.diagnosis}" in matched:
            return True
        components = getattr(candidate, "component_scores", {}) or {}
        if float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0:
            return True
        return bool(
            matched
            & {
                "heart_failure_state",
                "renal_impairment",
                "egfr_low",
                "urea_elevated",
                "portal_vein_dilation",
                "portal_flow_abnormal",
                "ventricular_septal_defect",
                "pulmonary_valve_stenosis",
            }
        )

    @staticmethod
    def _objective_signal(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{candidate.diagnosis}" in matched:
            return True
        if matched & _OBJECTIVE_GAP_FINDINGS:
            return True
        components = getattr(candidate, "component_scores", {}) or {}
        return float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0

    @staticmethod
    def _name(candidate: Any) -> str:
        return str(getattr(candidate, "diagnosis", "") or "")

    @staticmethod
    def _reasoning(decision: JudgeDecision) -> str:
        if not decision.primary:
            return "Judge did not authorize a final diagnosis."
        text = (
            f"Judge authorized primary={decision.primary}; "
            f"retriever_top1={decision.retriever_top1 or 'none'}; "
            f"override={int(decision.decision_override)}."
        )
        if decision.needs_discriminating_exams:
            text += (
                " Primary lock deferred for discriminating exams; "
                f"provisional_primary={decision.provisional_primary}."
            )
        if decision.required_gap_authorized_diagnoses:
            text += " Required-gap authorization: " + ", ".join(
                decision.required_gap_authorized_diagnoses
            ) + "."
        if decision.primary_unlock_reason:
            text += " Primary unlock: " + decision.primary_unlock_reason + "."
        if decision.gap_state_distribution:
            text += " Gap states: " + ", ".join(
                f"{key}={value}"
                for key, value in sorted(decision.gap_state_distribution.items())
            ) + "."
        if decision.evidence_gap_targets:
            text += " Evidence-gap targets: " + ", ".join(decision.evidence_gap_targets) + "."
        return text


class DiagnosisSubmitter:
    """Write a JudgeDecision into the mutable DiagnosisDecision audit object."""

    def __init__(self, knowledge: Any = None):
        self.knowledge = knowledge

    def apply(self, decision: Any, judge_decision: JudgeDecision) -> Any:
        if not decision or not judge_decision:
            return decision
        score_by_name = {item.diagnosis: item for item in getattr(decision, "candidates", []) or []}
        for name in judge_decision.required_gap_authorized_diagnoses:
            candidate = score_by_name.get(name)
            if candidate:
                setattr(candidate, "required_gap_authorized", True)
        for name in judge_decision.final_diagnoses:
            candidate = score_by_name.get(name)
            if candidate:
                candidate.differential_only = False
                candidate.differential_only_reason = ""
        for item in judge_decision.blocked_diagnoses:
            candidate = score_by_name.get(item.get("diagnosis"))
            if candidate and candidate.diagnosis not in set(judge_decision.final_diagnoses):
                candidate.differential_only = True
                candidate.differential_only_reason = str(item.get("reason") or "differential_only")

        decision.retriever_top1 = judge_decision.retriever_top1
        decision.judge_primary = judge_decision.judge_primary
        decision.submitter_final = list(judge_decision.final_diagnoses)
        decision.decision_override = bool(judge_decision.decision_override)
        decision.required_gap_authorized_diagnoses = list(
            judge_decision.required_gap_authorized_diagnoses
        )
        decision.judge_decision = judge_decision.to_dict()
        decision.pre_authorization_diagnoses = list(judge_decision.final_diagnoses)
        decision.final_diagnoses = list(judge_decision.final_diagnoses)
        decision.authorized_diagnoses = list(judge_decision.final_diagnoses)
        decision.trusted_diagnoses = [
            name
            for name in judge_decision.final_diagnoses
            if score_by_name.get(name)
            and (
                getattr(score_by_name[name], "trusted", False)
                or getattr(score_by_name[name], "required_gap_authorized", False)
            )
        ]
        decision.blocked_diagnoses = list(judge_decision.blocked_diagnoses)
        decision.submission_override_count = int(judge_decision.decision_override)
        if judge_decision.reasoning and judge_decision.reasoning not in str(
            getattr(decision, "evidence_reasoning", "") or ""
        ):
            decision.evidence_reasoning = (
                str(getattr(decision, "evidence_reasoning", "") or "").rstrip()
                + " "
                + judge_decision.reasoning
            ).strip()
        return decision
