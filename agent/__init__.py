"""
Agent 包。

使用懒加载避免 python -m agent.agent 时的 RuntimeWarning。
"""

__all__ = [
    "MyDoctorAgent",
    "DefectDetector",
    "PolicyStore",
    "ExamStrategyAgent",
    "InquiryStrategyAgent",
    "QualityAgent",
    "TreatmentStrategyAgent",
    "StructuralDiagnosisAgent",
    "EvidenceDiagnosisEngine",
    "ClinicalEvidenceNormalizer",
    "EvidenceAgent",
    "EvidenceGraph",
    "EvidenceBundle",
    "Observation",
    "CandidateGenerator",
    "CandidatePool",
    "CandidateSource",
    "DiagnosisDecisionEngine",
    "DiagnosticKnowledgeBase",
    "DiagnosisDecision",
    "OpenWorldDiagnosisResolver",
    "DiagnosisResolution",
    "DiagnosisCritic",
    "CriticDecision",
    "TreatmentSafetyGate",
    "DiagnosticLearningStore",
    "DoctorAgentMemory",
    "WorkingCaseMemory",
    "MemoryItem",
    "MemoryConfig",
    "HybridRAGRetriever",
    "HybridRAGConfig",
    "RagChunk",
    "ShadowReplay",
    "DiagnosticReplay",
    "heuristic_plan_score",
]


def __getattr__(name: str):
    """懒加载：仅在访问时才导入符号。"""
    if name == "MyDoctorAgent":
        from .agent import MyDoctorAgent
        return MyDoctorAgent
    if name == "DefectDetector":
        from .critic import DefectDetector
        return DefectDetector
    if name == "PolicyStore":
        from .policy_store import PolicyStore
        return PolicyStore
    if name == "ExamStrategyAgent":
        from .exam_strategy import ExamStrategyAgent
        return ExamStrategyAgent
    if name == "InquiryStrategyAgent":
        from .inquiry_strategy import InquiryStrategyAgent
        return InquiryStrategyAgent
    if name == "QualityAgent":
        from .qc import QualityAgent
        return QualityAgent
    if name == "TreatmentStrategyAgent":
        from .treatment_strategy import TreatmentStrategyAgent
        return TreatmentStrategyAgent
    if name == "StructuralDiagnosisAgent":
        from .structural_diagnosis import StructuralDiagnosisAgent
        return StructuralDiagnosisAgent
    if name == "EvidenceDiagnosisEngine":
        from .evidence_engine import EvidenceDiagnosisEngine
        return EvidenceDiagnosisEngine
    if name in ("ClinicalEvidenceNormalizer", "EvidenceAgent", "EvidenceGraph", "EvidenceBundle", "Observation"):
        from . import clinical_evidence as _clinical_evidence
        return getattr(_clinical_evidence, name)
    if name in ("CandidateGenerator", "CandidatePool", "CandidateSource"):
        from . import candidate_generator as _candidate_generator
        return getattr(_candidate_generator, name)
    if name in ("DiagnosisDecisionEngine", "DiagnosticKnowledgeBase", "DiagnosisDecision"):
        from . import diagnosis_engine as _diagnosis_engine
        return getattr(_diagnosis_engine, name)
    if name in ("OpenWorldDiagnosisResolver", "DiagnosisResolution"):
        from . import diagnosis_resolver as _diagnosis_resolver
        return getattr(_diagnosis_resolver, name)
    if name in ("DiagnosisCritic", "CriticDecision"):
        from . import diagnosis_critic as _diagnosis_critic
        return getattr(_diagnosis_critic, name)
    if name == "TreatmentSafetyGate":
        from .treatment_safety import TreatmentSafetyGate
        return TreatmentSafetyGate
    if name == "DiagnosticLearningStore":
        from .diagnostic_learning import DiagnosticLearningStore
        return DiagnosticLearningStore
    if name in ("DoctorAgentMemory", "WorkingCaseMemory", "MemoryItem", "MemoryConfig"):
        from . import memory_system as _memory_system
        return getattr(_memory_system, name)
    if name in ("HybridRAGRetriever", "HybridRAGConfig", "RagChunk"):
        from . import rag_retriever as _rag_retriever
        return getattr(_rag_retriever, name)
    if name in ("ShadowReplay", "DiagnosticReplay", "heuristic_plan_score"):
        from . import replay as _replay
        return getattr(_replay, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
