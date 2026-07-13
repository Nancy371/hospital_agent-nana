"""
医学知识库模块 (KnowledgeBase)
================================
将 data/ref_data/ 下的三份静态字典从"格式白名单"升级为"主动召回的知识库"。

三大能力：
1. 症状→鉴别诊断倒排索引 (recall_diseases_by_symptoms)
   - 基于疾病名/科室做启发式关键词展开，构建症状-疾病倒排。
   - _think() 前做一次规则召回，缩小 LLM 搜索空间。
2. 检查项规范化 (normalize_examinations / score_examination_coverage)
   - 将 LLM 输出的检查名对齐到 examinations_catalog；
   - 对遗漏的高价值检查给出补充建议；
   - 对无效/幻觉检查给出剔除。
3. 三源 RAG 上下文组装 (build_rag_context)
   - 命中的疾病定义 + 典型检 + 归属科室，供反思/诊断阶段注入 prompt。

设计原则：
- 纯本地字典，零外部依赖，符合平台"不依赖 docker-compose"约束；
- 全部是启发式规则 + 集合运算，性能 O(N)，可安全内嵌到主循环；
- 失败降级：字典缺失或加载失败时返回空结果，不影响主流程。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============ 症状→疾病 启发式关键词映射 ============
# 由于 diseases_catalog.json 只有 name/department/icd10，没有典型症状字段，
# 这里以疾病名为核心做人工启发式关键词展开。命中即召回。
# 后续可通过训练积累的 memory 反向丰富此表（feedback loop）。
_DISEASE_SYMPTOM_HINTS: Dict[str, List[str]] = {
    "肺炎": ["咳嗽", "咳痰", "发热", "胸痛", "呼吸困难", "气促"],
    "支气管炎": ["咳嗽", "咳痰", "喘息"],
    "支气管哮喘": ["喘息", "气促", "呼吸困难", "胸闷"],
    "慢性阻塞性肺疾病": ["咳嗽", "咳痰", "呼吸困难", "气促"],
    "肺结核": ["咳嗽", "咳痰", "咯血", "低热", "盗汗", "消瘦"],
    "肺癌": ["咳嗽", "咯血", "胸痛", "消瘦"],
    "高血压": ["头晕", "头痛", "耳鸣"],
    "冠心病": ["胸痛", "胸闷", "心悸", "气短"],
    "心力衰竭": ["气促", "呼吸困难", "下肢水肿", "乏力"],
    "心律失常": ["心悸", "头晕", "晕厥"],
    "心肌梗死": ["胸痛", "胸闷", "大汗", "呼吸困难"],
    "胃炎": ["腹痛", "恶心", "呕吐", "反酸", "嗳气"],
    "胃溃疡": ["上腹痛", "反酸", "嗳气", "黑便"],
    "十二指肠溃疡": ["空腹痛", "夜间痛", "反酸"],
    "肝硬化": ["腹胀", "乏力", "黄疸", "腹水"],
    "胆囊炎": ["右上腹痛", "发热", "恶心", "呕吐"],
    "胰腺炎": ["上腹痛", "腰背放射痛", "恶心", "呕吐"],
    "肠梗阻": ["腹痛", "腹胀", "呕吐", "停止排气排便"],
    "阑尾炎": ["右下腹痛", "发热", "恶心", "呕吐"],
    "糖尿病": ["多饮", "多尿", "多食", "消瘦", "乏力"],
    "甲状腺功能亢进": ["心悸", "多汗", "消瘦", "手抖", "易怒"],
    "甲状腺功能减退": ["乏力", "怕冷", "水肿", "体重增加"],
    "痛风": ["关节痛", "关节红肿", "夜间痛"],
    "贫血": ["乏力", "头晕", "面色苍白", "心悸"],
    "白血病": ["发热", "出血", "贫血", "乏力"],
    "脑梗死": ["肢体无力", "言语不清", "口角歪斜", "头晕"],
    "脑出血": ["头痛", "呕吐", "意识障碍", "肢体无力"],
    "癫痫": ["抽搐", "意识丧失", "口吐白沫"],
    "帕金森病": ["手抖", "肢体僵硬", "行动迟缓"],
    "骨折": ["外伤", "疼痛", "肿胀", "活动受限"],
    "腰椎间盘突出": ["腰痛", "下肢放射痛", "麻木"],
    "骨关节炎": ["关节痛", "晨僵", "活动受限"],
    "泌尿系感染": ["尿频", "尿急", "尿痛", "发热"],
    "肾结石": ["腰痛", "血尿", "肾绞痛"],
    "前列腺增生": ["尿频", "尿急", "排尿困难", "夜尿多"],
    "过敏性鼻炎": ["鼻塞", "喷嚏", "流涕", "鼻痒"],
    "中耳炎": ["耳痛", "听力下降", "耳鸣", "发热"],
    "白内障": ["视力下降", "视物模糊"],
    "青光眼": ["眼痛", "视力下降", "头痛", "呕吐"],
    "湿疹": ["皮疹", "瘙痒"],
    "荨麻疹": ["皮疹", "瘙痒", "风团"],
    "抑郁症": ["情绪低落", "兴趣减退", "失眠", "疲乏"],
    "焦虑症": ["紧张", "心悸", "失眠", "易怒"],
    "类风湿关节炎": ["关节痛", "晨僵", "对称性关节肿胀"],
    "系统性红斑狼疮": ["皮疹", "关节痛", "发热", "乏力"],
    "上呼吸道感染": ["咳嗽", "咽痛", "流涕", "鼻塞", "发热"],
    "急性胃肠炎": ["腹痛", "腹泻", "呕吐", "发热"],
    "颈椎病": ["颈痛", "上肢麻木", "头晕"],
}


# ============ 疾病→典型检查项 启发式映射 ============
_DISEASE_EXAM_HINTS: Dict[str, List[str]] = {
    "肺炎": ["血常规", "C反应蛋白", "胸部CT", "胸部X线"],
    "支气管炎": ["血常规", "胸部X线"],
    "支气管哮喘": ["肺功能", "血常规"],
    "慢性阻塞性肺疾病": ["肺功能", "胸部CT", "血气分析"],
    "肺结核": ["胸部CT", "血沉"],
    "肺癌": ["胸部CT", "肿瘤标志物", "支气管镜"],
    "高血压": ["心电图", "血脂", "肾功能"],
    "冠心病": ["心电图", "心肌酶谱", "血脂", "超声心动图"],
    "心力衰竭": ["超声心动图", "心电图", "胸部X线"],
    "心律失常": ["心电图", "动态心电图", "电解质"],
    "心肌梗死": ["心电图", "心肌酶谱", "D-二聚体"],
    "胃炎": ["胃镜", "血常规"],
    "胃溃疡": ["胃镜", "便常规"],
    "十二指肠溃疡": ["胃镜", "便常规"],
    "肝硬化": ["肝功能", "腹部B超", "凝血功能"],
    "胆囊炎": ["血常规", "腹部B超", "肝功能"],
    "胰腺炎": ["血常规", "腹部CT"],
    "肠梗阻": ["腹部CT", "腹部B超", "电解质"],
    "阑尾炎": ["血常规", "腹部B超", "C反应蛋白"],
    "糖尿病": ["血糖", "尿常规", "肾功能"],
    "甲状腺功能亢进": ["甲状腺功能"],
    "甲状腺功能减退": ["甲状腺功能"],
    "痛风": ["肾功能", "血常规"],
    "贫血": ["血常规"],
    "白血病": ["血常规", "血沉"],
    "脑梗死": ["头颅CT", "头颅MRI", "凝血功能"],
    "脑出血": ["头颅CT"],
    "癫痫": ["脑电图", "头颅MRI"],
    "帕金森病": ["头颅MRI"],
    "骨折": ["胸部X线", "腰椎X线", "颈椎X线"],
    "腰椎间盘突出": ["腰椎X线"],
    "骨关节炎": ["骨密度"],
    "泌尿系感染": ["尿常规", "血常规"],
    "肾结石": ["尿常规", "腹部B超", "腹部CT"],
    "前列腺增生": ["腹部B超"],
    "过敏性鼻炎": [],
    "中耳炎": ["血常规"],
    "白内障": ["眼底检查", "视野检查"],
    "青光眼": ["眼底检查", "视野检查"],
    "湿疹": [],
    "荨麻疹": [],
    "抑郁症": [],
    "焦虑症": [],
    "类风湿关节炎": ["血沉", "C反应蛋白", "血常规"],
    "系统性红斑狼疮": ["血常规", "尿常规", "肾功能"],
    "上呼吸道感染": ["血常规"],
    "急性胃肠炎": ["血常规", "便常规", "电解质"],
    "颈椎病": ["颈椎X线", "头颅MRI"],
}


_DISEASE_PRIORITY = {
    name: idx for idx, name in enumerate([
        "心肌梗死", "脑出血", "脑梗死", "肺炎", "阑尾炎", "胰腺炎", "肠梗阻",
        "胆囊炎", "肾结石", "泌尿系感染", "冠心病", "心力衰竭", "心律失常",
        "慢性阻塞性肺疾病", "支气管哮喘", "肺结核", "肺癌", "高血压",
        "糖尿病", "白血病", "贫血", "上呼吸道感染", "支气管炎",
    ])
}


class KnowledgeBase:
    """静态医学知识库 —— 症状召回 + 检查规范化 + RAG 上下文组装。"""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.diseases: List[Dict[str, Any]] = []
        self.examinations: List[Dict[str, Any]] = []
        self.departments: List[Dict[str, Any]] = []
        self.disease_profiles: List[Dict[str, Any]] = []
        # 索引结构
        self._disease_by_name: Dict[str, Dict[str, Any]] = {}
        self._exam_by_name: Dict[str, Dict[str, Any]] = {}
        self._profile_by_name: Dict[str, Dict[str, Any]] = {}
        self._alias_to_profile_name: Dict[str, str] = {}
        self._symptom_to_diseases: Dict[str, Set[str]] = {}
        self._load()
        self._build_indices()

    # ---------- 加载 ----------
    def _load(self) -> None:
        try:
            for filename, attr in [
                ("diseases_catalog.json", "diseases"),
                ("examinations_catalog.json", "examinations"),
                ("departments.json", "departments"),
            ]:
                path = os.path.join(self.ref_dir, filename)
                if not os.path.exists(path):
                    logger.warning(f"[Knowledge] 参考字典缺失: {path}")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 字典根字段名与属性名一致
                setattr(self, attr, data.get(attr, []))
            self.disease_profiles = []
            for profiles_file in ("disease_profiles.json", "disease_profiles_extra.json"):
                profiles_path = os.path.join(self.ref_dir, profiles_file)
                if os.path.exists(profiles_path):
                    with open(profiles_path, "r", encoding="utf-8") as f:
                        profiles_data = json.load(f)
                    self.disease_profiles.extend(profiles_data.get("profiles", []))
            logger.info(
                f"[Knowledge] 加载完成: 疾病={len(self.diseases)}, "
                f"检查={len(self.examinations)}, 科室={len(self.departments)}, "
                f"画像={len(self.disease_profiles)}"
            )
        except Exception as e:
            logger.warning(f"[Knowledge] 加载失败: {e}")

    def _build_indices(self) -> None:
        """构建症状→疾病倒排索引与名称索引。"""
        self._disease_by_name = {d.get("name"): d for d in self.diseases if d.get("name")}
        self._exam_by_name = {e.get("name"): e for e in self.examinations if e.get("name")}
        self._profile_by_name = {
            p.get("name"): p for p in self.disease_profiles if p.get("name")
        }
        self._alias_to_profile_name = {}
        for profile in self.disease_profiles:
            name = profile.get("name")
            if not name:
                continue
            self._alias_to_profile_name[name] = name
            for alias in profile.get("aliases", []) or []:
                if alias:
                    self._alias_to_profile_name[str(alias)] = name

        # 症状倒排：从 _DISEASE_SYMPTOM_HINTS 反向展开
        for disease_name, symptom_list in _DISEASE_SYMPTOM_HINTS.items():
            if disease_name not in self._disease_by_name:
                continue  # 仅索引字典中真实存在的疾病
            for sym in symptom_list:
                self._symptom_to_diseases.setdefault(sym, set()).add(disease_name)
        # 疾病画像提供更细的症状与红旗信号。
        for profile in self.disease_profiles:
            disease_name = profile.get("name")
            if not disease_name:
                continue
            for sym in (profile.get("common_symptoms") or []) + (profile.get("red_flags") or []):
                if sym:
                    self._symptom_to_diseases.setdefault(str(sym), set()).add(disease_name)

    # ---------- 症状→疾病 召回 ----------
    def recall_diseases_by_symptoms(
        self, symptoms: List[str], top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """基于症状列表召回候选疾病，按命中数量排序。

        Args:
            symptoms: 症状关键词列表（可能是"咳嗽/发热"等短文本）
            top_k: 返回候选数

        Returns:
            [{name, department, icd10, hit_count, matched_symptoms}] 按 hit_count 降序
        """
        if not symptoms:
            return []

        # 用子串匹配增强健壮性（"干咳" ↔ "咳嗽"）
        hits: Dict[str, Dict[str, Any]] = {}
        for sym in symptoms:
            sym_key = str(sym).strip()
            if not sym_key:
                continue
            for indexed_sym, disease_set in self._symptom_to_diseases.items():
                if indexed_sym in sym_key or sym_key in indexed_sym:
                    for d_name in disease_set:
                        rec = hits.setdefault(
                            d_name,
                            {"name": d_name, "hit_count": 0, "matched_symptoms": []},
                        )
                        rec["hit_count"] += 1
                        if indexed_sym not in rec["matched_symptoms"]:
                            rec["matched_symptoms"].append(indexed_sym)

        # 附上疾病元信息并排序
        results = []
        for d_name, rec in hits.items():
            base = self._disease_by_name.get(d_name, {})
            results.append(
                {
                    "name": d_name,
                    "department": base.get("department", ""),
                    "icd10": base.get("icd10", ""),
                    "hit_count": rec["hit_count"],
                    "matched_symptoms": rec["matched_symptoms"],
                }
            )
        results.sort(
            key=lambda x: (-x["hit_count"], _DISEASE_PRIORITY.get(x["name"], 999), x["name"])
        )
        return results[:top_k]

    def normalize_diagnosis(self, name: str) -> Optional[str]:
        """将别名/近似诊断名对齐到标准疾病目录或疾病画像名。"""
        if not name:
            return None
        raw = str(name).strip()
        if raw in self._disease_by_name:
            return raw
        if raw in self._alias_to_profile_name:
            return self._alias_to_profile_name[raw]

        # 子串匹配：优先标准目录，其次画像别名。
        for standard in self._disease_by_name:
            if standard and (standard in raw or raw in standard):
                return standard
        for alias, standard in self._alias_to_profile_name.items():
            if alias and (alias in raw or raw in alias):
                return standard
        return None

    @staticmethod
    def _candidate_name(item: Any) -> Optional[str]:
        """从字符串或 LLM 返回的候选疾病对象中提取疾病名。"""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("disease", "diagnosis", "name"):
                if item.get(key):
                    return str(item[key])
        return None

    def _normalize_candidate_names(self, candidate_diseases: Optional[List[Any]]) -> List[str]:
        names: List[str] = []
        for item in candidate_diseases or []:
            name = self._candidate_name(item)
            if not name:
                continue
            names.append(self.normalize_diagnosis(name) or name)
        return list(dict.fromkeys(names))

    def get_disease_profile(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """按标准名或别名获取疾病画像。"""
        standard = self.normalize_diagnosis(disease_name) or disease_name
        return self._profile_by_name.get(standard)

    def recall_disease_profiles(
        self,
        symptoms: List[str],
        candidate_diseases: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """召回疾病画像，兼顾显式候选疾病和症状命中。"""
        scores: Dict[str, Dict[str, Any]] = {}

        for disease in self._normalize_candidate_names(candidate_diseases):
            standard = self.normalize_diagnosis(disease)
            if standard and standard in self._profile_by_name:
                rec = scores.setdefault(
                    standard,
                    {"profile": self._profile_by_name[standard], "score": 0, "matched_symptoms": []},
                )
                rec["score"] += 3

        symptom_text = " ".join(str(s) for s in symptoms or [])
        for profile in self.disease_profiles:
            name = profile.get("name")
            if not name:
                continue
            rec = scores.setdefault(
                name,
                {"profile": profile, "score": 0, "matched_symptoms": []},
            )
            for sym in (profile.get("common_symptoms") or []) + (profile.get("red_flags") or []):
                sym = str(sym)
                if sym and (sym in symptom_text or any(sym in str(s) or str(s) in sym for s in symptoms or [])):
                    rec["score"] += 1
                    if sym not in rec["matched_symptoms"]:
                        rec["matched_symptoms"].append(sym)

        ranked = [
            {
                **rec["profile"],
                "matched_symptoms": rec["matched_symptoms"],
                "hit_score": rec["score"],
            }
            for rec in scores.values()
            if rec["score"] > 0
        ]
        ranked.sort(
            key=lambda item: (
                -item.get("hit_score", 0),
                _DISEASE_PRIORITY.get(item.get("name"), 999),
                item.get("name", ""),
            )
        )
        return ranked[:top_k]

    def get_required_exams(
        self,
        candidate_diseases: Optional[List[str]] = None,
        symptoms: Optional[List[str]] = None,
        include_optional: bool = False,
    ) -> List[str]:
        """根据候选疾病/症状返回标准化后的必查检查。"""
        candidate_names = self._normalize_candidate_names(candidate_diseases or [])
        if candidate_names:
            profiles = [
                self._profile_by_name[name]
                for name in candidate_names
                if name in self._profile_by_name
            ]
        else:
            profiles = self.recall_disease_profiles(
                symptoms=symptoms or [],
                candidate_diseases=[],
                top_k=5,
            )
        items: List[str] = []
        for profile in profiles:
            items.extend(profile.get("required_exams") or [])
            if include_optional:
                items.extend(profile.get("optional_exams") or [])
        normalized, _ = self.normalize_examinations(items)
        return list(dict.fromkeys(normalized))

    def build_clinical_context(
        self,
        symptoms: List[str],
        candidate_diseases: Optional[List[str]] = None,
        max_profiles: int = 4,
    ) -> str:
        """返回疾病画像上下文：关键问诊、红旗、必查检查、鉴别诊断和治疗原则。"""
        profiles = self.recall_disease_profiles(
            symptoms=symptoms,
            candidate_diseases=candidate_diseases,
            top_k=max_profiles,
        )
        if not profiles:
            return ""

        lines = ["【疾病画像库】请优先关注以下高频/高危疾病画像："]
        for idx, profile in enumerate(profiles, 1):
            lines.append(f"{idx}. {profile.get('name')}（{profile.get('department', '')}）")
            if profile.get("matched_symptoms"):
                lines.append(f"   命中症状: {', '.join(profile['matched_symptoms'][:6])}")
            if profile.get("red_flags"):
                lines.append(f"   红旗信号: {', '.join(profile['red_flags'][:6])}")
            if profile.get("key_questions"):
                lines.append(f"   关键问诊: {', '.join(profile['key_questions'][:5])}")
            if profile.get("required_exams"):
                lines.append(f"   必查检查: {', '.join(profile['required_exams'])}")
            if profile.get("differential_diagnoses"):
                lines.append(f"   重点鉴别: {', '.join(profile['differential_diagnoses'][:6])}")
            if profile.get("avoid_mistakes"):
                lines.append(f"   易错提醒: {', '.join(profile['avoid_mistakes'][:3])}")
        return "\n".join(lines)

    # ---------- 检查项规范化与覆盖度打分 ----------
    def normalize_examinations(
        self, exam_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """将 LLM 输出的检查名对齐到 examinations_catalog。

        Returns:
            (valid_names, invalid_names): 分别为字典命中的检查名与幻觉/无法匹配的名称
        """
        valid, invalid = [], []
        catalog_names = list(self._exam_by_name.keys())
        for raw in exam_names or []:
            name = str(raw).strip()
            if not name:
                continue
            if name in self._exam_by_name:
                valid.append(name)
                continue
            # 子串匹配（"胸片" ↔ "胸部X线"）
            matched = None
            for cname in catalog_names:
                if cname in name or name in cname:
                    matched = cname
                    break
            if matched:
                valid.append(matched)
            else:
                invalid.append(name)
        return list(dict.fromkeys(valid)), list(dict.fromkeys(invalid))

    def score_examination_coverage(
        self, ordered_exams: List[str], candidate_diseases: List[str]
    ) -> Dict[str, Any]:
        """基于候选疾病评估已选检查的覆盖度。

        Args:
            ordered_exams: 医生已的检查列表（规范化后）
            candidate_diseases: 候选疾病名列表

        Returns:
            {coverage, recommended_supplements, redundant}
        """
        recommended: Set[str] = set()
        for d in candidate_diseases or []:
            standard = self.normalize_diagnosis(d) or d
            profile = self.get_disease_profile(standard)
            profile_exams = profile.get("required_exams", []) if profile else []
            for e in list(_DISEASE_EXAM_HINTS.get(standard, [])) + list(profile_exams):
                recommended.add(e)

        ordered_set = set(ordered_exams or [])
        supplements = sorted(recommended - ordered_set)
        # 冗余判断：已选但不在任何候选疾病典型检查列表内
        redundant = sorted([e for e in ordered_set if recommended and e not in recommended])

        coverage = (
            len(recommended & ordered_set) / len(recommended) if recommended else 0.0
        )
        return {
            "coverage": round(coverage, 3),
            "recommended_supplements": supplements,
            "redundant": redundant,
            "expected_total": len(recommended),
            "hit_count": len(recommended & ordered_set),
        }

    # ---------- 三源 RAG 上下文组装 ----------
    def build_rag_context(
        self,
        symptoms: List[str],
        candidate_diseases: Optional[List[str]] = None,
        max_diseases: int = 5,
    ) -> str:
        """将命中的疾病定义 + 典型检查 + 归属科室拼装为可注入 prompt 的文本。

        Args:
            symptoms: 症状列表（用于召回）
            candidate_diseases: 显式指定的候选疾病名，若给出则跳过症状召回
            max_diseases: 上限

        Returns:
            自然语言描述文本，可直接插入 prompt system 段
        """
        # 若未显式给出候选疾病，则用症状召回
        candidate_names = self._normalize_candidate_names(candidate_diseases)
        if candidate_names:
            picks = [
                {"name": d, **(self._disease_by_name.get(d, {}))}
                for d in candidate_names[:max_diseases]
                if d in self._disease_by_name or d in self._profile_by_name
            ]
        else:
            picks = self.recall_diseases_by_symptoms(symptoms, top_k=max_diseases)

        clinical_context = self.build_clinical_context(
            symptoms=symptoms or [],
            candidate_diseases=candidate_names,
            max_profiles=max_diseases,
        )

        if not picks and not clinical_context:
            return ""

        lines = ["【医学知识库参考】以下是基于当前症状的候选疾病及其典型检查（仅供参考，勿盲从）："]
        for i, d in enumerate(picks, 1):
            name = d.get("name")
            dept = d.get("department", "")
            icd = d.get("icd10", "")
            exams = _DISEASE_EXAM_HINTS.get(name, [])
            matched = d.get("matched_symptoms", [])
            head = f"  {i}. {name}"
            if dept:
                head += f" ({dept})"
            if icd:
                head += f" ICD-10:{icd}"
            if matched:
                head += f" | 匹配症状: {', '.join(matched)}"
            lines.append(head)
            if exams:
                lines.append(f"     典型检查: {', '.join(exams)}")
        if clinical_context:
            lines.append(clinical_context)
        return "\n".join(lines)

    # ---------- 辅助 ----------
    def get_department_by_disease(self, disease_name: str) -> str:
        rec = self._disease_by_name.get(disease_name, {})
        return rec.get("department", "")

    def is_valid_diagnosis(self, disease_name: str) -> bool:
        """诊断名是否在标准疾病目录中。"""
        return bool(disease_name and disease_name in self._disease_by_name)

    def get_disease_catalog_names(self) -> List[str]:
        """返回标准疾病目录名称。"""
        return list(self._disease_by_name.keys())

    def suggest_diagnoses(
        self,
        symptoms: Optional[List[str]] = None,
        candidate_diseases: Optional[List[Any]] = None,
        top_k: int = 3,
    ) -> List[str]:
        """基于候选诊断和症状给出标准疾病名兜底建议。"""
        suggestions: List[str] = []
        for name in self._normalize_candidate_names(candidate_diseases or []):
            standard = self.normalize_diagnosis(name)
            if standard and self.is_valid_diagnosis(standard):
                suggestions.append(standard)

        if symptoms:
            for rec in self.recall_diseases_by_symptoms(symptoms, top_k=top_k):
                name = rec.get("name")
                if name and self.is_valid_diagnosis(name):
                    suggestions.append(name)
            for profile in self.recall_disease_profiles(symptoms, top_k=top_k):
                name = self.normalize_diagnosis(profile.get("name", ""))
                if name and self.is_valid_diagnosis(name):
                    suggestions.append(name)

        return list(dict.fromkeys(suggestions))[:top_k]

    def is_valid_examination(self, name: str) -> bool:
        return name in self._exam_by_name

    def get_examination_catalog_names(self) -> List[str]:
        return list(self._exam_by_name.keys())
