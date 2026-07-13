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
    "DoctorAgentMemory",
    "WorkingCaseMemory",
    "MemoryItem",
    "MemoryConfig",
    "HybridRAGRetriever",
    "HybridRAGConfig",
    "RagChunk",
    "ShadowReplay",
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
    if name in ("DoctorAgentMemory", "WorkingCaseMemory", "MemoryItem", "MemoryConfig"):
        from . import memory_system as _memory_system
        return getattr(_memory_system, name)
    if name in ("HybridRAGRetriever", "HybridRAGConfig", "RagChunk"):
        from . import rag_retriever as _rag_retriever
        return getattr(_rag_retriever, name)
    if name in ("ShadowReplay", "heuristic_plan_score"):
        from . import replay as _replay
        return getattr(_replay, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
