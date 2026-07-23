"""Candidate pool generation for evidence-first diagnosis ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .clinical_evidence import EvidenceBundle, EvidenceGraph, Observation
from .disease_retrieval import DiseaseRetriever


@dataclass
class CandidateSource:
    raw_name: str
    canonical_name: str
    source: str
    prior: float = 0.0
    evidence_links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePool:
    items: List[CandidateSource] = field(default_factory=list)
    name_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_candidates: List[str] = field(default_factory=list)
    disease_categories: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        raw_name: Any,
        canonical_name: Any,
        source: str,
        prior: float = 0.0,
        evidence_links: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        canonical = str(canonical_name or "").strip()
        raw = str(raw_name or canonical).strip()
        if not canonical:
            return
        self.items.append(
            CandidateSource(
                raw_name=raw,
                canonical_name=canonical,
                source=str(source or "unknown"),
                prior=max(0.0, min(1.0, float(prior or 0.0))),
                evidence_links=list(dict.fromkeys(str(item) for item in (evidence_links or []) if str(item))),
                metadata=dict(metadata or {}),
            )
        )

    def priors(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for item in self.items:
            result[item.canonical_name] = max(result.get(item.canonical_name, 0.0), item.prior)
        return result

    def sources_by_name(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.items:
            grouped.setdefault(item.canonical_name, []).append(item.to_dict())
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "name_resolutions": list(self.name_resolutions),
            "unresolved_candidates": list(self.unresolved_candidates),
            "disease_categories": list(self.disease_categories),
        }


class CandidateGenerator:
    """Union candidates from symptoms, structured evidence, LLM, RAG, and memory."""

    def __init__(self, knowledge: Any, resolver: Any):
        self.knowledge = knowledge
        self.resolver = resolver
        self.disease_retriever = DiseaseRetriever(knowledge, resolver)

    def generate(
        self,
        evidence_graph: Optional[EvidenceGraph] = None,
        llm_result: Optional[Dict[str, Any]] = None,
        rag_chunks: Optional[Sequence[Dict[str, Any]]] = None,
        memory_hits: Optional[Sequence[Dict[str, Any]]] = None,
        evidence: Optional[EvidenceBundle] = None,
    ) -> CandidatePool:
        bundle = evidence or _bundle_from_graph(evidence_graph)
        pool = CandidatePool()
        self._from_llm(pool, llm_result or {})
        self._from_rag(pool, rag_chunks or [])
        self._from_memory(pool, memory_hits or [])
        self._from_disease_retriever(pool, bundle)
        self._from_evidence(pool, bundle)
        return pool

    def _from_llm(self, pool: CandidatePool, llm_result: Dict[str, Any]) -> None:
        resolutions = self.resolver.resolve_result(llm_result or {})
        pool.name_resolutions = [item.to_dict() for item in resolutions]
        pool.unresolved_candidates = [item.raw_name for item in resolutions if not item.resolved]
        for index, item in enumerate(resolutions):
            if not item.canonical_name:
                continue
            rank_prior = max(0.65, 1.0 - index * 0.10)
            prior = rank_prior * item.confidence * item.model_confidence
            pool.add(
                item.raw_name,
                item.canonical_name,
                "llm",
                prior=prior,
                metadata={"method": item.method, "parent_name": item.parent_name},
            )

    def _from_rag(self, pool: CandidatePool, rag_chunks: Sequence[Dict[str, Any]]) -> None:
        for chunk in rag_chunks:
            if chunk.get("type") != "disease_profile":
                continue
            resolution = self.resolver.resolve(chunk.get("title"))
            if not resolution.canonical_name:
                continue
            try:
                prior = float(chunk.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                prior = 0.0
            pool.add(
                chunk.get("title"),
                resolution.canonical_name,
                "rag",
                prior=prior,
                metadata={"chunk_id": chunk.get("id"), "chunk_type": chunk.get("type")},
            )

    def _from_memory(self, pool: CandidatePool, memory_hits: Sequence[Dict[str, Any]]) -> None:
        for hit in memory_hits:
            names = (
                hit.get("expected_diagnoses")
                or hit.get("diagnosis")
                or hit.get("diagnoses")
                or hit.get("submitted_diagnoses")
                or []
            )
            if isinstance(names, str):
                names = [names]
            try:
                prior = float(hit.get("score", hit.get("similarity", 0.45)) or 0.45)
            except (TypeError, ValueError):
                prior = 0.45
            for raw in names:
                resolution = self.resolver.resolve(raw)
                if resolution.canonical_name:
                    pool.add(raw, resolution.canonical_name, "memory", prior=min(0.75, prior))

    def _from_disease_retriever(self, pool: CandidatePool, evidence: EvidenceBundle) -> None:
        hits, categories = self.disease_retriever.retrieve(evidence, top_k=20)
        pool.disease_categories = [item.to_dict() for item in categories]
        for hit in hits:
            pool.add(
                hit.diagnosis,
                hit.diagnosis,
                "disease_retriever",
                prior=hit.score,
                evidence_links=hit.evidence_links,
                metadata={
                    "category": hit.category,
                    **dict(hit.metadata or {}),
                },
            )

    def _from_evidence(self, pool: CandidatePool, evidence: EvidenceBundle) -> None:
        observations = evidence.observations if evidence else []
        for name, entry in self.knowledge.entries.items():
            matched = []
            weight = 0.0
            for spec in entry.get("supporting_evidence", []) or []:
                hits = _matching_observations(spec, observations)
                if not hits:
                    continue
                spec_weight = float(spec.get("weight", 0.2) or 0.2)
                confidence = max(item.confidence for item in hits)
                weight += spec_weight * confidence
                matched.extend(item.finding for item in hits)
            if matched:
                pool.add(
                    name,
                    name,
                    "evidence",
                    prior=min(0.85, max(0.20, weight / 1.5)),
                    evidence_links=list(dict.fromkeys(matched)),
                )


def _bundle_from_graph(graph: Optional[EvidenceGraph]) -> EvidenceBundle:
    if graph is None:
        return EvidenceBundle()
    if getattr(graph, "bundle", None):
        return graph.bundle
    observations: List[Observation] = []
    for item in getattr(graph, "observations", []) or []:
        if isinstance(item, Observation):
            observations.append(item)
        elif isinstance(item, dict):
            try:
                observations.append(Observation(**{k: v for k, v in item.items() if k in Observation.__dataclass_fields__}))
            except (TypeError, ValueError):
                continue
    return EvidenceBundle(observations)


def _matching_observations(spec: Dict[str, Any], observations: Sequence[Observation]) -> List[Observation]:
    return [
        item for item in observations
        if item.polarity == "positive" and _observation_matches(spec, item)
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
