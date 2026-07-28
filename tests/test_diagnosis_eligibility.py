import unittest
from types import SimpleNamespace

from agent.diagnosis_eligibility import (
    ANCHORS_SATISFIED,
    DEFERRED,
    DIFFERENTIAL_ONLY,
    EXCLUDED,
    NEEDS_ANCHOR,
    PATTERN_CONTRADICTED,
    PRIMARY_ELIGIBLE,
    DiagnosisEligibilityGate,
)


def candidate(**overrides):
    data = {
        "diagnosis": "candidate",
        "diagnosis_type": "disease",
        "matched_evidence": ["symptom:signal"],
        "core_matched_evidence": [],
        "diagnostic_matched_evidence": [],
        "generic_matched_evidence": [],
        "required_gaps": [],
        "required_met": True,
        "hard_contradiction": False,
        "hard_contradicted_evidence": [],
        "unresolved_evidence_conflict": False,
        "differential_only": False,
        "differential_only_reason": "",
        "source_prior": 0.4,
        "coverage_score": 0.5,
        "core_explanatory_coverage": 0.5,
        "diagnostic_evidence_score": 0.0,
        "core_evidence_score": 0.0,
        "residual_core_evidence_count": 0,
        "component_scores": {},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class DiagnosisEligibilityGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = DiagnosisEligibilityGate()

    def test_missing_vitamin_d_anchor_defers_rickets_for_workup(self):
        rickets = candidate(
            diagnosis="vitamin_d_deficiency_rickets",
            diagnosis_type="metabolic",
            required_met=False,
            matched_evidence=["bone_pain", "waddling_gait", "alp_elevated", "hypocalcemia"],
            core_matched_evidence=["bone_pain", "alp_elevated", "hypocalcemia"],
            required_gaps=["vitamin_d_low|bone_deformity"],
            source_prior=0.55,
            coverage_score=0.62,
            core_explanatory_coverage=0.58,
        )

        result = self.gate.evaluate(rickets)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        self.assertIn("vitamin_d_low|bone_deformity", result.missing_required_anchors)

    def test_closed_vitamin_d_anchor_is_primary_eligible(self):
        rickets = candidate(
            diagnosis="vitamin_d_deficiency_rickets",
            diagnosis_type="metabolic",
            required_met=True,
            matched_evidence=["vitamin_d_low", "hypocalcemia", "alp_elevated", "bone_pain"],
            core_matched_evidence=["hypocalcemia", "alp_elevated", "bone_pain"],
            diagnostic_matched_evidence=["vitamin_d_low"],
            core_evidence_score=0.72,
            diagnostic_evidence_score=0.62,
        )

        result = self.gate.evaluate(rickets)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)
        self.assertIn("vitamin_d_low", result.satisfied_required_anchors)

    def test_missing_pericardial_anchor_does_not_exclude_candidate(self):
        tb_pericarditis = candidate(
            diagnosis="tuberculous_pericarditis",
            diagnosis_type="etiology",
            required_met=False,
            matched_evidence=["cough", "fever", "dyspnea"],
            required_gaps=["pericardial_effusion", "tb_microbiology_positive"],
            source_prior=0.2,
            coverage_score=0.18,
            core_explanatory_coverage=0.05,
        )

        result = self.gate.evaluate(tb_pericarditis)

        self.assertIn(result.status, {DEFERRED, DIFFERENTIAL_ONLY})
        self.assertNotEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertNotEqual(result.status, EXCLUDED)

    def test_hard_contradiction_excludes_candidate(self):
        tb_pericarditis = candidate(
            diagnosis="tuberculous_pericarditis",
            required_met=False,
            hard_contradiction=True,
            hard_contradicted_evidence=["normal_pericardium"],
            matched_evidence=["cough", "fever"],
            required_gaps=["pericardial_effusion", "tb_microbiology_positive"],
        )

        result = self.gate.evaluate(tb_pericarditis)

        self.assertEqual(result.status, EXCLUDED)
        self.assertIn("normal_pericardium", result.blockers)

    def test_downstream_state_with_poor_global_explanation_is_differential_only(self):
        low_magnesium = candidate(
            diagnosis="low_magnesium",
            diagnosis_type="metabolic",
            required_met=True,
            matched_evidence=["magnesium_depletion"],
            diagnostic_matched_evidence=["magnesium_depletion"],
            coverage_score=0.25,
            core_explanatory_coverage=0.2,
            diagnostic_evidence_score=0.2,
            core_evidence_score=0.1,
            residual_core_evidence_count=4,
        )

        result = self.gate.evaluate(low_magnesium)

        self.assertEqual(result.status, DIFFERENTIAL_ONLY)

    def test_prostatitis_without_urinary_or_prostate_anchor_is_deferred(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=["fever", "acute_course", "cough", "dyspnea", "bronchopneumonia"],
            core_matched_evidence=["fever", "acute_course"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
            core_evidence_score=0.20,
            diagnostic_evidence_score=0.0,
        )

        result = self.gate.evaluate(prostatitis)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)

    def test_pyuria_alone_is_not_prostatitis_anchor(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=["pyuria"],
            core_matched_evidence=["pyuria"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.45,
            core_explanatory_coverage=0.36,
        )

        result = self.gate.evaluate(prostatitis)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        self.assertIn(
            "acute_bacterial_prostatitis_requires_urinary_or_prostate_anchor",
            result.missing_required_anchors,
        )

    def test_negative_urine_pattern_downgrades_pyuria_support(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=[
                "pyuria",
                "urine_culture_no_growth",
                "leukocyte_esterase_negative",
                "nitrite_negative",
            ],
            core_matched_evidence=["pyuria"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.45,
            core_explanatory_coverage=0.36,
        )

        result = self.gate.evaluate(prostatitis)

        self.assertEqual(result.status, DIFFERENTIAL_ONLY)
        self.assertEqual(result.reason, PATTERN_CONTRADICTED)
        self.assertIn("urine_culture_no_growth", result.blockers)
        self.assertEqual(result.evidence_pattern_matches[0]["role"], "negative_pattern")

    def test_confirmed_bacterial_prostatitis_pattern_is_primary_eligible(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=[
                "prostate_tenderness",
                "dysuria",
                "pyuria",
                "urine_culture_positive",
            ],
            core_matched_evidence=["prostate_tenderness", "dysuria"],
            diagnostic_matched_evidence=["urine_culture_positive"],
            required_gaps=[],
            core_evidence_score=0.62,
            diagnostic_evidence_score=0.55,
        )

        result = self.gate.evaluate(prostatitis)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)

    def test_pulmonary_cryptococcosis_without_fungal_anchor_is_deferred(self):
        crypto = candidate(
            diagnosis="肺隐球菌病",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cough", "fever", "acute_course", "dyspnea"],
            core_matched_evidence=["cough", "fever"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
        )

        result = self.gate.evaluate(crypto)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)

    def test_pulmonary_cryptococcosis_with_cryptococcal_anchor_is_primary_eligible(self):
        crypto = candidate(
            diagnosis="肺隐球菌病",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cryptococcal_antigen_positive", "pulmonary_nodule"],
            core_matched_evidence=["cryptococcal_antigen_positive"],
            diagnostic_matched_evidence=["cryptococcal_antigen_positive"],
            required_gaps=[],
            diagnostic_evidence_score=0.55,
            core_evidence_score=0.55,
        )

        result = self.gate.evaluate(crypto)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)

    def test_mycoplasma_pneumonia_without_pathogen_anchor_is_deferred(self):
        mycoplasma = candidate(
            diagnosis="支原体肺炎",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cough", "fever", "dyspnea", "pneumonia_infiltrate"],
            core_matched_evidence=["cough", "fever"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
        )

        result = self.gate.evaluate(mycoplasma)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)

    def test_mycoplasma_pneumonia_with_pathogen_anchor_is_primary_eligible(self):
        mycoplasma = candidate(
            diagnosis="支原体肺炎",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["mycoplasma_naat_positive", "interstitial_infiltrate"],
            core_matched_evidence=["mycoplasma_naat_positive"],
            diagnostic_matched_evidence=["mycoplasma_naat_positive"],
            required_gaps=[],
            diagnostic_evidence_score=0.55,
            core_evidence_score=0.55,
        )

        result = self.gate.evaluate(mycoplasma)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)


if __name__ == "__main__":
    unittest.main()
