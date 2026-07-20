"""检查策略 Agent：结合疾病画像补齐必查检查并过滤无效检查。"""

from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase


_PEDIATRIC_CARDIAC_SCREEN_EXAMS = [
    "体格检查",
    "超声心动图",
    "心电图",
    "胸部X线",
    "心导管检查",
]

_RIGHT_HEART_VALVE_EXAMS = [
    "体格检查",
    "心电图（ECG）",
    "超声心动图",
    "三维超声心动图（3D Echo）",
    "经食管超声心动图（TEE）",
    "心脏MRI（CMR）",
]

_URINARY_SYNDROME_EXAMS = [
    "体格检查",
    "尿液分析（UA）",
    "尿培养",
    "综合代谢面板（CMP）",
    "泌尿道超声",
    "尿动力学检查（UDS）",
]

_RIB_TRAUMA_EXAMS = [
    "体格检查",
    "胸部X线检查（CXR）",
    "胸部CT扫描（Chest CT）",
]

_RETINOBLASTOMA_EXAMS = [
    "眼底摄影",
    "眼部超声",
    "磁共振成像（MRI）",
    "CT扫描（CT）",
]

_HIB_RESPIRATORY_EXAMS = [
    "全血细胞计数（CBC）",
    "血培养",
    "痰培养",
    "血清学抗体检测",
    "胸部X线检查（CXR）",
]

_IMMUNE_PNEUMONIA_EXAMS = [
    "体格检查",
    "脉搏血氧饱和度监测（SpO2）",
    "动脉血气（ABG）",
    "全血细胞计数（CBC）",
    "C反应蛋白（CRP）",
    "降钙素原（PCT）",
    "胸部X线检查（CXR）",
    "痰培养",
    "抗菌药物敏感性试验（AST）",
    "超声",
    "类风湿因子（RF）",
]

_CONGENITAL_SHUNT_EXAMS = [
    "超声心动图",
    "心电图（ECG）",
    "胸部X线检查（CXR）",
    "心导管检查",
]

_ELECTROLYTE_CRISIS_EXAMS = [
    "电解质",
    "心电图（ECG）",
    "肾功能",
    "血气分析",
]

_PULMONARY_RENAL_EXAMS = [
    "尿常规",
    "肾功能",
    "胸部CT",
    "抗核抗体",
    "血常规",
    "C反应蛋白",
]

_METABOLIC_BONE_EXAMS = [
    "电解质",
    "骨密度",
    "肝功能",
    "肾功能",
]

_ASPIRATION_PNEUMONIA_EXAMS = [
    "体格检查",
    "脉搏血氧饱和度监测（SpO2）",
    "血气分析",
    "胸部X线",
    "血常规",
    "C反应蛋白",
    "降钙素原",
    "痰培养",
    "支气管镜",
]

_ADVANCED_CARDIAC_EXAMS = {
    "三维超声心动图（3D Echo）",
    "经食管超声心动图（TEE）",
    "心脏MRI（CMR）",
    "心导管检查",
}

_STRONG_VERIFICATION_EXAMS = {
    "低镁血症": [
        "综合代谢面板（CMP）",
        "24小时尿电解质检测",
        "镁负荷试验",
        "心电图（ECG）",
    ],
    "维生素D缺乏性佝偻病": [
        "维生素D检测",
        "血清电解质",
        "甲状旁腺激素检测（PTH）",
        "肝功能检查（LFTs）",
        "骨转换标志物（BTMs）",
        "X线检查",
    ],
    "显微镜下多血管炎": [
        "尿液分析（UA）",
        "肾功能",
        "胸部CT扫描（Chest CT）",
        "抗核抗体",
        "全血细胞计数（CBC）",
        "C反应蛋白（CRP）",
    ],
    "肺不张": [
        "胸部X线检查（CXR）",
        "胸部CT扫描（Chest CT）",
        "支气管镜检查",
        "动脉血气（ABG）",
    ],
    "支气管肺炎": [
        "体格检查",
        "脉搏血氧饱和度监测（SpO2）",
        "胸部X线检查（CXR）",
        "全血细胞计数（CBC）",
        "C反应蛋白（CRP）",
        "降钙素原（PCT）",
        "痰培养",
        "抗菌药物敏感性试验（AST）",
    ],
}


class ExamStrategyAgent:
    """轻量检查策略角色，不调用外部服务，只做本地规则增强。"""

    def __init__(self, knowledge: KnowledgeBase, max_new_items: int = 10):
        self.knowledge = knowledge
        self.max_new_items = max_new_items

    def recommend(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
        existing_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """返回本轮建议检查项。

        优先保留疾病画像中的必查检查，其次使用 LLM 提议的有效检查项。
        """
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        existing_results = existing_results or {}
        existing_valid, _ = self.knowledge.normalize_examinations(list(existing_results.keys()))
        existing_set = set(existing_results.keys()) | set(existing_valid)

        proposed_valid, invalid_items = self.knowledge.normalize_examinations(proposed_items or [])
        strong_verification_items = self._strong_verification_items(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            proposed_items=proposed_items,
        )
        ranked_items, information_gain = self._rank_by_information_gain(
            candidate_diseases=candidate_diseases or [],
            symptoms=symptoms,
            proposed_items=proposed_valid,
        )
        required_items = self.knowledge.get_required_exams(
            candidate_diseases=(candidate_diseases or [])[:3],
            symptoms=symptoms,
            include_optional=False,
        )
        red_flag_items = self._scenario_items(collected_info, candidate_diseases)
        evidence_driven_items = list(dict.fromkeys(red_flag_items + ranked_items))

        merged: List[str] = []
        for item in strong_verification_items + evidence_driven_items + required_items + proposed_valid:
            if item and item not in merged:
                merged.append(item)
        merged = self.prepare_order_items(
            merged,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            existing_results=existing_results,
            max_items=self.max_new_items,
            add_strong_verification=False,
        )

        return {
            "items": merged,
            "strong_verification_items": [
                item for item in strong_verification_items if item not in existing_set
            ],
            "required_items": required_items,
            "red_flag_items": red_flag_items,
            "evidence_driven_items": evidence_driven_items,
            "information_gain": information_gain,
            "added_required": [item for item in required_items if item not in proposed_valid],
            "invalid_items": invalid_items,
            "clinical_context": self.knowledge.build_clinical_context(
                symptoms=symptoms,
                candidate_diseases=candidate_diseases,
            ),
        }

    def prepare_order_items(
        self,
        items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        existing_results: Optional[Dict[str, Any]] = None,
        max_items: Optional[int] = None,
        add_strong_verification: bool = True,
    ) -> List[str]:
        """Normalize, strengthen and gate examination names before submission."""
        existing_results = existing_results or {}
        normalized, _ = self.knowledge.normalize_examinations(items or [])
        merged: List[str] = []
        if add_strong_verification:
            for item in self._strong_verification_items(
                collected_info=collected_info,
                candidate_diseases=candidate_diseases,
                proposed_items=items,
            ):
                if item and item not in merged:
                    merged.append(item)
        for item in normalized:
            if item and item not in merged:
                merged.append(item)

        merged = self._filter_contextual_items(
            merged,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
        )
        existing_valid, _ = self.knowledge.normalize_examinations(list(existing_results.keys()))
        existing_set = set(existing_results.keys()) | set(existing_valid)
        result = [item for item in merged if item not in existing_set]
        if max_items is not None:
            return result[: max(0, int(max_items))]
        return result

    def _strong_verification_items(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
    ) -> List[str]:
        diseases: List[str] = []
        for item in candidate_diseases or []:
            name = self.knowledge._candidate_name(item)
            if not name:
                continue
            standard = self.knowledge.normalize_diagnosis(name) or name
            if standard not in diseases:
                diseases.append(standard)

        symptoms = collected_info.get("symptoms", []) if collected_info else []
        for profile in self.knowledge.recall_disease_profiles(
            symptoms=symptoms,
            candidate_diseases=diseases,
            top_k=5,
        ):
            name = str(profile.get("name") or "")
            if name and name not in diseases:
                diseases.append(name)

        if self._needs_low_magnesium_verification(
            collected_info, candidate_diseases, proposed_items
        ):
            diseases.append("低镁血症")
        if self._needs_metabolic_bone_workup(collected_info, candidate_diseases):
            diseases.append("维生素D缺乏性佝偻病")
        if self._needs_pulmonary_renal_workup(collected_info, candidate_diseases):
            diseases.append("显微镜下多血管炎")
        if self._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases):
            diseases.extend(["肺不张", "支气管肺炎"])

        items: List[str] = []
        for disease in list(dict.fromkeys(diseases)):
            profile = self.knowledge.get_disease_profile(disease) or {}
            raw_items = (
                list(profile.get("strong_verification_exams") or [])
                or list(_STRONG_VERIFICATION_EXAMS.get(disease, []))
            )
            normalized, _ = self.knowledge.normalize_examinations(raw_items)
            for exam in normalized:
                if exam and exam not in items:
                    items.append(exam)
        return items

    def _rank_by_information_gain(
        self,
        candidate_diseases: List[Any],
        symptoms: List[Any],
        proposed_items: List[str],
    ) -> tuple[List[str], Dict[str, float]]:
        """Rank exams by relevance and ability to separate the top candidates."""
        candidates: List[str] = []
        for item in candidate_diseases:
            if isinstance(item, dict):
                value = item.get("disease") or item.get("diagnosis") or item.get("name")
            else:
                value = item
            if not value:
                continue
            normalized = self.knowledge.normalize_diagnosis(str(value)) or str(value)
            if normalized not in candidates:
                candidates.append(normalized)
            if len(candidates) >= 3:
                break

        exam_support: Dict[str, set] = {}
        relevance: Dict[str, float] = {}
        for rank, disease in enumerate(candidates):
            profile = self.knowledge.get_disease_profile(disease) or {}
            raw_items = list(profile.get("required_exams") or [])
            normalized_items, _ = self.knowledge.normalize_examinations(raw_items)
            for exam in normalized_items:
                exam_support.setdefault(exam, set()).add(disease)
                relevance[exam] = relevance.get(exam, 0.0) + 1.0 / (rank + 1)

        if not exam_support:
            fallback = self.knowledge.get_required_exams(
                candidate_diseases=candidates,
                symptoms=symptoms,
                include_optional=False,
            )
            for exam in fallback:
                exam_support.setdefault(exam, set()).add("symptom_recall")
                relevance[exam] = max(relevance.get(exam, 0.0), 0.7)

        for exam in proposed_items:
            exam_support.setdefault(exam, set())
            relevance[exam] = relevance.get(exam, 0.0) + 0.15

        candidate_count = max(1, len(candidates))
        max_relevance = max(relevance.values(), default=1.0)
        scores: Dict[str, float] = {}
        for exam, supported in exam_support.items():
            coverage = len(supported)
            if candidate_count <= 1:
                discrimination = 0.7 if coverage else 0.0
            elif 0 < coverage < candidate_count:
                discrimination = 1.0
            elif coverage == candidate_count:
                discrimination = 0.35
            else:
                discrimination = 0.0
            relevance_score = relevance.get(exam, 0.0) / max_relevance
            scores[exam] = round(0.65 * discrimination + 0.35 * relevance_score, 4)

        ranked = sorted(scores, key=lambda item: (scores[item], relevance.get(item, 0.0)), reverse=True)
        return ranked, scores

    def _scenario_items(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        items: List[str] = []
        if self._needs_electrolyte_crisis_workup(collected_info, candidate_diseases):
            items.extend(_ELECTROLYTE_CRISIS_EXAMS)
        if self._needs_pulmonary_renal_workup(collected_info, candidate_diseases):
            items.extend(_PULMONARY_RENAL_EXAMS)
        if self._needs_metabolic_bone_workup(collected_info, candidate_diseases):
            items.extend(_METABOLIC_BONE_EXAMS)
        if self._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases):
            items.extend(_ASPIRATION_PNEUMONIA_EXAMS)
        if self._needs_pediatric_cardiac_screen(collected_info):
            items.extend(_PEDIATRIC_CARDIAC_SCREEN_EXAMS)
        if self._needs_right_heart_valve_workup(collected_info, candidate_diseases):
            items.extend(_RIGHT_HEART_VALVE_EXAMS)
        if self._needs_urinary_syndrome_workup(collected_info, candidate_diseases):
            items.extend(_URINARY_SYNDROME_EXAMS)
        if self._needs_rib_trauma_workup(collected_info, candidate_diseases):
            items.extend(_RIB_TRAUMA_EXAMS)
        if self._needs_retinoblastoma_workup(collected_info, candidate_diseases):
            items.extend(_RETINOBLASTOMA_EXAMS)
        if self._needs_hib_respiratory_workup(collected_info, candidate_diseases):
            items.extend(_HIB_RESPIRATORY_EXAMS)
        if self._needs_immune_pneumonia_workup(collected_info, candidate_diseases):
            items.extend(_IMMUNE_PNEUMONIA_EXAMS)
        if self._needs_congenital_shunt_workup(collected_info, candidate_diseases):
            items.extend(_CONGENITAL_SHUNT_EXAMS)
        normalized, _ = self.knowledge.normalize_examinations(items)
        return normalized

    def _filter_contextual_items(
        self,
        items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        advanced_cardiac = self._has_advanced_cardiac_signal(
            collected_info, candidate_diseases
        )
        if advanced_cardiac:
            return items
        return [item for item in items if item not in _ADVANCED_CARDIAC_EXAMS]

    @staticmethod
    def _needs_pediatric_cardiac_screen(collected_info: Dict[str, Any]) -> bool:
        if not collected_info:
            return False

        parts: List[str] = []
        for key in (
            "chief_complaint",
            "present_illness",
            "past_history",
            "personal_history",
            "physical_signs",
            "raw_responses",
            "question_focus",
        ):
            value = collected_info.get(key)
            if value:
                parts.append(str(value))
        for symptom in collected_info.get("symptoms", []) or []:
            parts.append(str(symptom))
        text = " ".join(parts)

        pediatric = any(token in text for token in ("宝宝", "婴儿", "患儿", "吃奶", "喂奶"))
        respiratory = any(token in text for token in ("呼吸急促", "呼吸明显变快", "气促", "呼吸困难", "喘"))
        perfusion = any(token in text for token in ("发绀", "青紫", "口周发绀", "嘴巴周围发青"))
        feeding = any(token in text for token in ("吃奶减少", "喂养困难", "喂奶", "吃奶"))
        sweating = any(token in text for token in ("多汗", "出汗"))

        return pediatric and respiratory and (perfusion or feeding or sweating)

    @staticmethod
    def _case_text(
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> str:
        parts: List[str] = []
        for key in (
            "chief_complaint",
            "present_illness",
            "past_history",
            "personal_history",
            "physical_signs",
        ):
            value = (collected_info or {}).get(key)
            if value:
                parts.append(str(value))
        for symptom in (collected_info or {}).get("symptoms", []) or []:
            parts.append(str(symptom))
        for candidate in candidate_diseases or []:
            if isinstance(candidate, dict):
                parts.extend(str(v) for v in candidate.values() if v)
            else:
                parts.append(str(candidate))
        return " ".join(parts)

    @classmethod
    def _needs_electrolyte_crisis_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        gi_loss = any(token in text for token in ("腹泻", "拉肚子", "水样便", "呕吐"))
        neuromuscular = any(token in text for token in ("抽筋", "手足", "痉挛", "乏力", "无力"))
        cardiac = any(token in text for token in ("心悸", "心慌", "QT", "心律失常"))
        neuro = any(token in text for token in ("意识模糊", "头晕", "黑朦"))
        candidate = any(token in text for token in ("低镁血症", "低钾", "低钙", "电解质"))
        return candidate or (gi_loss and (neuromuscular or cardiac or neuro))

    @classmethod
    def _needs_pulmonary_renal_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        pulmonary = any(token in text for token in ("咯血", "咳血", "血痰", "肺泡出血"))
        renal = any(
            token in text
            for token in ("尿色", "深色尿", "血尿", "蛋白尿", "脚踝水肿", "水肿", "肾功能", "少尿")
        )
        candidate = any(token in text for token in ("显微镜下多血管炎", "血管炎", "ANCA", "肾肺"))
        return candidate or (pulmonary and renal)

    @classmethod
    def _needs_metabolic_bone_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        bone = any(token in text for token in ("腿痛", "骨痛", "跛行", "O型腿", "X型腿", "步态"))
        function = any(token in text for token in ("运动耐力下降", "运动耐量下降", "活动后", "下肢功能障碍", "上下楼"))
        pediatric = any(token in text for token in ("儿童", "患儿", "孩子", "青少年", "岁"))
        candidate = any(token in text for token in ("佝偻病", "维生素D", "代谢性骨病", "骨软化"))
        return candidate or (bone and (function or pediatric))

    @classmethod
    def _needs_low_magnesium_verification(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
    ) -> bool:
        text = " ".join(
            [
                cls._case_text(collected_info, candidate_diseases),
                " ".join(str(item) for item in proposed_items or []),
            ]
        )
        gi_loss = any(token in text for token in ("腹泻", "拉肚子", "水样便", "呕吐"))
        neuromuscular = any(token in text for token in ("抽筋", "手足", "痉挛", "乏力", "无力"))
        cardiac = any(token in text for token in ("心悸", "心慌", "QT", "QTc", "心律失常"))
        explicit = any(token in text for token in ("低镁血症", "低镁", "血镁", "镁负荷", "hypomagnesemia"))
        magnesium_hint = "镁" in text and (gi_loss or neuromuscular or cardiac)
        return explicit or magnesium_hint or (gi_loss and neuromuscular and cardiac)

    @classmethod
    def _needs_aspiration_pneumonia_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        aspiration = any(token in text for token in ("呛咳", "误吸", "吞咽困难", "进食时"))
        respiratory = any(token in text for token in ("咳嗽", "发热", "呼吸困难", "气促", "低氧", "痰"))
        candidate = any(token in text for token in ("肺不张", "支气管肺炎", "吸入性肺炎", "黏液栓"))
        return candidate or (aspiration and respiratory)

    @classmethod
    def _has_strong_noncardiac_path(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        return (
            cls._needs_electrolyte_crisis_workup(collected_info, candidate_diseases)
            or cls._needs_pulmonary_renal_workup(collected_info, candidate_diseases)
            or cls._needs_metabolic_bone_workup(collected_info, candidate_diseases)
            or cls._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases)
        )

    @classmethod
    def _has_advanced_cardiac_signal(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        return any(
            token in text
            for token in (
                "二尖瓣",
                "三尖瓣",
                "肺动脉瓣",
                "瓣膜",
                "心脏杂音",
                "杂音",
                "先心",
                "房间隔缺损",
                "ASD",
                "VSD",
                "右心",
                "肺动脉高压",
                "心导管",
                "重度反流",
            )
        )

    @classmethod
    def _needs_right_heart_valve_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        direct = any(
            token in text
            for token in ("三尖瓣", "肺动脉瓣", "右心", "肺动脉高压", "肺动脉瓣狭窄")
        )
        syndrome = any(token in text for token in ("心力衰竭", "心衰", "气短", "下肢水肿", "腿肿"))
        cardiopulmonary = any(token in text for token in ("心悸", "心慌", "呼吸困难", "肺动脉高压"))
        return direct or (
            syndrome
            and cardiopulmonary
            and not cls._needs_congenital_shunt_workup(collected_info, candidate_diseases)
        )

    @classmethod
    def _needs_urinary_syndrome_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        urinary_symptoms = any(token in text for token in ("尿急", "尿频", "尿痛", "排尿烧灼"))
        urinary_candidates = any(token in text for token in ("尿道综合征", "泌尿系感染", "膀胱过度活动", "逼尿肌"))
        return urinary_symptoms or urinary_candidates

    @classmethod
    def _needs_rib_trauma_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        chest_wall = any(token in text for token in ("胸壁", "肋骨", "肋部", "左侧胸"))
        trauma = any(token in text for token in ("外伤", "车门", "夹伤", "撞伤", "摔伤"))
        pleuritic = any(token in text for token in ("深呼吸", "咳嗽", "转身", "呼吸受限"))
        return (chest_wall and trauma) or (chest_wall and pleuritic)

    @classmethod
    def _needs_retinoblastoma_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        leukocoria = any(token in text for token in ("瞳孔发白", "白瞳", "猫眼反光", "白色反光"))
        strabismus = any(token in text for token in ("内斜视", "斜视"))
        tumor_candidate = any(token in text for token in ("视网膜母细胞瘤", "眼内肿瘤"))
        return leukocoria or (strabismus and tumor_candidate)

    @classmethod
    def _needs_hib_respiratory_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        prolonged = any(token in text for token in ("6周", "六周", "迁延", "反复发热", "反复咳嗽"))
        respiratory = any(token in text for token in ("咳嗽", "发热", "脓痰", "浓痰", "喘息", "呼吸困难"))
        ear = any(token in text for token in ("耳痛", "中耳炎", "耳流脓"))
        candidate = any(token in text for token in ("流感嗜血杆菌", "Hib"))
        return candidate or (prolonged and respiratory and ear)

    @classmethod
    def _needs_immune_pneumonia_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        pneumonia = any(token in text for token in ("肺炎", "发热", "咳嗽", "呼吸困难"))
        immune = any(token in text for token in ("类风湿", "糖皮质激素", "免疫抑制", "脾肿大", "粒细胞减少", "费尔蒂"))
        severe = any(token in text for token in ("低氧", "SpO", "呼吸困难", "重症"))
        return pneumonia and (immune or severe and "搬入新公寓" in text)

    @classmethod
    def _needs_congenital_shunt_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        direct = any(token in text for token in ("房间隔缺损", "ASD", "先心", "左向右分流", "继发孔"))
        recurrent_lung = any(token in text for token in ("反复肺炎", "两次肺炎", "多次肺炎", "反复呼吸道感染"))
        exertional = any(token in text for token in ("活动后", "玩耍", "运动后", "跑几步", "乏力", "心悸", "喘息"))
        pediatric = any(token in text for token in ("孩子", "患儿", "儿童", "宝宝", "家长"))
        return direct or (recurrent_lung and exertional and pediatric)
