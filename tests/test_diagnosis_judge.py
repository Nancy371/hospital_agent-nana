import unittest

import yaml

from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine


YAWS = "\u96c5\u53f8\u75c5"
ECZEMA = "\u6e7f\u75b9"
WATERPOX = "\u6c34\u75d8"
LEUKEMIA = "\u767d\u8840\u75c5"
PORTAL_HTN = "\u95e8\u9759\u8109\u9ad8\u538b"
TB = "\u80ba\u7ed3\u6838"
PNEUMONIA = "\u80ba\u708e"
BRONCHOPNEUMONIA = "\u652f\u6c14\u7ba1\u80ba\u708e"
LUNG_CANCER = "\u80ba\u764c"
MPA = "\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e"
URACHAL_CYST = "\u8110\u5c3f\u7ba1\u56ca\u80bf"
FRACTURE = "\u9aa8\u6298"
ZOSTER = "\u5e26\u72b6\u75b1\u75b9"
CONGENITAL_HEART = "\u5148\u5929\u6027\u5fc3\u810f\u75c5"
PULMONARY_STENOSIS = "\u80ba\u52a8\u8109\u74e3\u72ed\u7a84"
HEART_FAILURE = "\u5fc3\u529b\u8870\u7aed"
OHSS = "\u5375\u5de2\u8fc7\u5ea6\u523a\u6fc0\u7efc\u5408\u5f81"
PANCREATITIS = "\u80f0\u817a\u708e"
URETHRAL_SYNDROME = "\u5c3f\u9053\u7efc\u5408\u5f81"
AV_BLOCK_2 = "\u4e8c\u5ea6\u623f\u5ba4\u4f20\u5bfc\u963b\u6ede"
ARRHYTHMIA = "\u5fc3\u5f8b\u5931\u5e38"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate(
    name,
    score,
    *,
    required=True,
    diagnosis_type="disease",
    specificity=0.7,
    coverage=0.5,
    residual=0.4,
    core_coverage=None,
    residual_core=0,
    matched=None,
    gaps=None,
    parent="",
    core_score=0.0,
    diagnostic_score=0.0,
    generic_penalty=0.0,
):
    if core_coverage is None:
        core_coverage = coverage
    return CandidateScore(
        diagnosis=name,
        score=score,
        support_score=score,
        source_prior=0.5,
        explanation_score=coverage,
        coverage_score=coverage,
        residual_score=residual,
        explanatory_coverage=coverage,
        core_explanatory_coverage=core_coverage,
        residual_evidence_score=residual,
        residual_core_evidence_count=residual_core,
        explained_evidence=list(matched or ["symptom:signal"]),
        unexplained_core_evidence=[
            f"core_gap_{index + 1}" for index in range(max(0, int(residual_core)))
        ],
        contradiction_penalty=0.0,
        required_met=required,
        hard_contradiction=False,
        matched_evidence=list(matched or ["symptom:signal"]),
        core_matched_evidence=[
            item for item in list(matched or []) if item not in {"fever", "cough", "pain", "rash"}
        ],
        diagnostic_matched_evidence=[],
        core_evidence_score=core_score,
        diagnostic_evidence_score=diagnostic_score,
        generic_coverage_score=0.0,
        required_gaps=list(gaps or []),
        component_scores={
            "core_evidence_score": core_score,
            "diagnostic_evidence_score": diagnostic_score,
            "generic_parent_penalty": generic_penalty,
        },
        diagnosis_type=diagnosis_type,
        parent_diagnosis=parent,
        specificity=specificity,
    )


class DiagnosisJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")

    def run_candidates(self, candidates):
        decision = DiagnosisDecision(
            final_diagnoses=[],
            trusted_diagnoses=[],
            candidates=candidates,
            unexplained_evidence=[],
            confidence=0.0,
            margin=0.0,
            low_confidence=False,
        )
        return self.engine.judge_and_submit(decision)

    def test_explanatory_power_beats_required_met_with_core_residual(self):
        yaws = candidate(
            YAWS,
            0.52,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.74,
            residual=0.18,
            core_coverage=0.78,
            residual_core=0,
            matched=[
                "rash",
                "treponemal_skin_lesion",
                "periostitis",
                "tropical_exposure",
            ],
            gaps=["treponema_positive"],
        )
        eczema = candidate(
            ECZEMA,
            0.62,
            required=True,
            specificity=0.48,
            coverage=0.30,
            residual=0.70,
            core_coverage=0.20,
            residual_core=3,
            matched=["rash", "pruritus"],
        )
        decision = self.run_candidates([eczema, yaws])
        payload = decision.judge_decision
        self.assertEqual(decision.final_diagnoses, [YAWS])
        self.assertIn(YAWS, decision.required_gap_authorized_diagnoses)
        self.assertTrue(payload["explanation_score_changed_ranking"])
        self.assertEqual(payload["required_gap_state_by_candidate"][YAWS], "actionable_gap")
        self.assertEqual(payload["residual_core_evidence_count"], 0)

    def test_primary_lock_deferred_when_yaws_competes_with_waterpox(self):
        waterpox = candidate(
            WATERPOX,
            0.58,
            required=True,
            specificity=0.90,
            coverage=0.52,
            residual=0.38,
            matched=["vesicular_rash", "pruritus"],
        )
        yaws = candidate(
            YAWS,
            0.55,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.60,
            residual=0.30,
            matched=["treponemal_skin_lesion", "regional_lymphadenopathy"],
            gaps=["treponema_positive"],
        )
        leukemia = candidate(
            LEUKEMIA,
            0.44,
            required=True,
            specificity=0.78,
            coverage=0.30,
            residual=0.58,
            matched=["fever"],
        )
        decision = self.run_candidates([waterpox, yaws, leukemia])
        payload = decision.judge_decision
        self.assertEqual(payload["primary_status"], "deferred")
        self.assertTrue(payload["needs_discriminating_exams"])
        self.assertIn(YAWS, payload["differential_candidates"])
        self.assertIn("treponema_positive", payload["discriminating_findings"])
        self.assertIn(YAWS, decision.required_gap_authorized_diagnoses)

    def test_tb_is_not_locked_out_by_required_met_pneumonia(self):
        pneumonia = candidate(
            BRONCHOPNEUMONIA,
            0.58,
            required=True,
            specificity=0.72,
            coverage=0.46,
            residual=0.42,
            matched=["cough", "fever"],
        )
        tb = candidate(
            TB,
            0.54,
            required=False,
            diagnosis_type="etiology",
            specificity=0.92,
            coverage=0.68,
            residual=0.25,
            core_coverage=0.70,
            residual_core=0,
            matched=["hemoptysis", "night_sweats", "tuberculosis_exposure"],
            gaps=["afb_positive", "tb_naat_positive"],
        )
        lung_cancer = candidate(
            LUNG_CANCER,
            0.48,
            required=True,
            specificity=0.82,
            coverage=0.46,
            residual=0.44,
            matched=["hemoptysis"],
        )
        decision = self.run_candidates([pneumonia, tb, lung_cancer])
        payload = decision.judge_decision
        self.assertEqual(decision.final_diagnoses[0], TB)
        self.assertIn(TB, payload["differential_candidates"])
        self.assertIn(TB, decision.required_gap_authorized_diagnoses)

    def test_urachal_cyst_core_evidence_beats_urethral_syndrome(self):
        urethral = candidate(
            URETHRAL_SYNDROME,
            0.64,
            required=True,
            diagnosis_type="syndrome",
            specificity=0.58,
            coverage=0.38,
            residual=0.62,
            core_coverage=0.12,
            residual_core=3,
            matched=["dysuria", "urinary_frequency"],
            generic_penalty=0.8,
        )
        urachal = candidate(
            URACHAL_CYST,
            0.50,
            required=False,
            diagnosis_type="structural",
            specificity=0.93,
            coverage=0.66,
            residual=0.18,
            core_coverage=0.78,
            residual_core=0,
            matched=[
                "umbilical_discharge",
                "midline_suprapubic_pain",
                "urachal_remnant_pattern",
            ],
            gaps=["urachal_cyst_imaging"],
            core_score=0.86,
        )
        decision = self.run_candidates([urethral, urachal])
        self.assertEqual(decision.final_diagnoses[0], URACHAL_CYST)
        self.assertIn(URACHAL_CYST, decision.required_gap_authorized_diagnoses)
        self.assertTrue(decision.judge_decision["explanation_score_changed_ranking"])

    def test_av_block_core_evidence_beats_arrhythmia_parent(self):
        arrhythmia = candidate(
            ARRHYTHMIA,
            0.68,
            required=True,
            specificity=0.50,
            coverage=0.40,
            residual=0.55,
            core_coverage=0.18,
            residual_core=2,
            matched=["palpitation", "dizziness"],
            generic_penalty=0.75,
        )
        av_block = candidate(
            AV_BLOCK_2,
            0.52,
            required=True,
            diagnosis_type="structural",
            specificity=0.94,
            coverage=0.70,
            residual=0.16,
            core_coverage=0.82,
            residual_core=0,
            matched=["second_degree_av_block", "bradycardia", "presyncope"],
            parent=ARRHYTHMIA,
            core_score=0.74,
            diagnostic_score=0.52,
        )
        decision = self.run_candidates([arrhythmia, av_block])
        self.assertEqual(decision.final_diagnoses[0], AV_BLOCK_2)
        self.assertNotIn(ARRHYTHMIA, decision.final_diagnoses)

    def test_tb_mpa_lung_cancer_tasks_prioritize_special_discriminators(self):
        lung_cancer = candidate(
            LUNG_CANCER,
            0.62,
            required=True,
            specificity=0.86,
            coverage=0.50,
            residual=0.36,
            matched=["hemoptysis"],
        )
        mpa = candidate(
            MPA,
            0.59,
            required=False,
            diagnosis_type="systemic",
            specificity=0.92,
            coverage=0.62,
            residual=0.26,
            core_coverage=0.64,
            residual_core=1,
            matched=["hemoptysis", "microscopic_hematuria"],
            gaps=["anca_positive", "renal_impairment"],
        )
        tb = candidate(
            TB,
            0.56,
            required=False,
            diagnosis_type="etiology",
            specificity=0.92,
            coverage=0.68,
            residual=0.22,
            core_coverage=0.70,
            residual_core=0,
            matched=["hemoptysis", "night_sweats", "tuberculosis_exposure"],
            gaps=["afb_positive", "tb_naat_positive"],
        )
        decision = self.run_candidates([lung_cancer, mpa, tb])
        payload = decision.judge_decision
        tasks = payload["discriminating_exam_tasks"]
        exams = [item["exam"] for item in tasks]
        self.assertLessEqual(len(exams), 6)
        self.assertLess(
            exams.index("抗酸杆菌染色（AFB）"),
            exams.index("全血细胞计数（CBC）") if "全血细胞计数（CBC）" in exams else len(exams),
        )
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", exams)
        self.assertTrue(
            all(item["target_candidates"] for item in tasks)
        )
        self.assertGreaterEqual(
            sum(1 for item in tasks if len(item["target_candidates"]) >= 2),
            4,
        )

    def test_differential_pool_filters_cross_system_noise_from_yaws(self):
        waterpox = candidate(WATERPOX, 0.60, required=True, specificity=0.90, matched=["vesicular_rash"])
        yaws = candidate(
            YAWS,
            0.57,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.62,
            residual=0.28,
            matched=["treponemal_skin_lesion", "regional_lymphadenopathy"],
            gaps=["treponema_positive"],
        )
        leukemia = candidate(LEUKEMIA, 0.52, required=True, matched=["fever", "bone_pain"])
        portal = candidate(PORTAL_HTN, 0.55, required=True, diagnosis_type="systemic", matched=["portal_flow_abnormal"])
        decision = self.run_candidates([waterpox, yaws, portal, leukemia])
        payload = decision.judge_decision
        self.assertIn(YAWS, payload["differential_candidates"])
        self.assertIn(LEUKEMIA, payload["differential_candidates"])
        self.assertNotIn(PORTAL_HTN, payload["differential_candidates"])

    def test_top20_tail_high_specificity_candidate_enters_differential_pool(self):
        head = [
            candidate(
                f"generic_{i}",
                0.70 - i * 0.01,
                required=True,
                specificity=0.55,
                coverage=0.42,
                residual=0.44,
                matched=[f"symptom:generic{i}"],
            )
            for i in range(6)
        ]
        urachal = candidate(
            URACHAL_CYST,
            0.24,
            required=False,
            diagnosis_type="structural",
            specificity=0.93,
            coverage=0.28,
            residual=0.50,
            matched=["umbilical_discharge", "midline_suprapubic_cyst"],
            gaps=["urachal_cyst_imaging"],
        )
        decision = self.run_candidates(head + [urachal])
        payload = decision.judge_decision
        self.assertIn(URACHAL_CYST, payload["differential_candidates"])
        self.assertEqual(payload["differential_pool_source"].get(URACHAL_CYST), "top20_priority_tail")
        self.assertIn(URACHAL_CYST, payload["required_gap_state_by_candidate"])

    def test_hard_contradiction_still_blocks_gap_authorization(self):
        ohss = candidate(
            OHSS,
            0.70,
            required=False,
            diagnosis_type="systemic",
            specificity=0.94,
            coverage=0.8,
            residual=0.1,
            matched=["hemoconcentration"],
            gaps=["pelvic_ultrasound"],
        )
        ohss.hard_contradiction = True
        pancreatitis = candidate(PANCREATITIS, 0.52, required=True)
        decision = self.run_candidates([ohss, pancreatitis])
        self.assertEqual(decision.final_diagnoses, [PANCREATITIS])
        self.assertNotIn(OHSS, decision.required_gap_authorized_diagnoses)

    def test_direct_congenital_parent_beats_gap_child_even_without_parent_field(self):
        pulmonary = candidate(
            PULMONARY_STENOSIS,
            0.58,
            required=False,
            diagnosis_type="structural",
            specificity=0.92,
            coverage=0.64,
            residual=0.24,
            matched=["cyanosis", "right_ventricular_hypertrophy"],
            gaps=["pulmonary_valve_gradient"],
            parent=HEART_FAILURE,
        )
        congenital = candidate(
            CONGENITAL_HEART,
            0.47,
            required=True,
            diagnosis_type="structural",
            specificity=0.9,
            coverage=0.58,
            residual=0.30,
            matched=[f"diagnosis:{CONGENITAL_HEART}", "congenital_heart_defect"],
        )
        decision = self.run_candidates([pulmonary, congenital])
        self.assertEqual(decision.final_diagnoses, [CONGENITAL_HEART])

    def test_primary_unlock_reason_records_changed_preselection(self):
        previous = candidate(ECZEMA, 0.60, required=True, coverage=0.30, residual=0.70, residual_core=3)
        yaws = candidate(
            YAWS,
            0.56,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.74,
            residual=0.18,
            core_coverage=0.78,
            matched=["treponemal_skin_lesion", "periostitis"],
            gaps=["treponema_positive"],
        )
        judge_decision = self.engine.judge.judge([previous, yaws], preselected=[ECZEMA])
        self.assertEqual(judge_decision.primary, YAWS)
        self.assertTrue(judge_decision.primary_unlock_reason)


if __name__ == "__main__":
    unittest.main()
