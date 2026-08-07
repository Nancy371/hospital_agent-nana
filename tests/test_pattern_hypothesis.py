import unittest

import yaml

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.pattern_hypothesis import PatternHypothesisVerifier, ThinkingSnapshot


RAD_PNEUMONITIS = "\u653e\u5c04\u6027\u80ba\u708e"
MITRAL_REGURGITATION = "\u4e8c\u5c16\u74e3\u53cd\u6d41"
PAVM_ENTITY_ID = "D100055"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def radiation_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        [
            Observation(
                "thoracic_radiotherapy",
                "patient_reported_observation",
                confidence=0.94,
                information_value=0.9,
            ),
            Observation(
                "post_radiotherapy_time_window",
                "disease_agnostic_deterministic_relation",
                confidence=0.9,
                information_value=0.86,
            ),
            Observation(
                "ground_glass_opacity",
                "imaging_result",
                confidence=0.92,
                information_value=0.92,
            ),
            Observation(
                "dyspnea",
                "patient_reported_observation",
                confidence=0.88,
                information_value=0.62,
            ),
        ]
    )


def radiation_hypothesis():
    return {
        "pattern_hypothesis_id": "PH_RAD_001",
        "pattern_name": "post_thoracic_radiotherapy_lung_injury_pattern",
        "pattern_type": "temporal_causal_multievidence",
        "evidence_bindings": [
            {
                "evidence_id": "thoracic_radiotherapy",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "exposure",
            },
            {
                "evidence_id": "post_radiotherapy_time_window",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "temporal_relation",
            },
            {
                "evidence_id": "ground_glass_opacity",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "imaging_or_objective_finding",
            },
            {
                "evidence_id": "dyspnea",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "organ_manifestation",
            },
        ],
        "relations": [
            {
                "type": "temporal_after",
                "from": "thoracic_radiotherapy",
                "to": "dyspnea",
            }
        ],
        "suggested_diseases": [
            {
                "name": RAD_PNEUMONITIS,
                "canonical_id": "radiation_pneumonitis",
                "hypothesis_confidence": 0.86,
            }
        ],
        "missing_evidence_requests": [
            {"target_evidence": "infection_exclusion", "importance": "supportive"}
        ],
        "model_confidence": 0.86,
    }


class PatternHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")
        cls.verifier = PatternHypothesisVerifier(cls.engine.knowledge, config=cls.config)

    def test_missing_source_evidence_is_rejected(self):
        evidence = radiation_evidence()
        payload = radiation_hypothesis()
        payload["evidence_bindings"][0]["evidence_id"] = "missing_evidence"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
        )
        rejected = context["pattern_verification_results"][0]
        self.assertEqual(rejected["verification_status"], "rejected")
        self.assertIn("unsupported_source_evidence", rejected["rejection_reasons"])
        self.assertTrue(
            any(
                item["entity_id"] == "D100058"
                and item["pattern_hypothesis_id"].startswith("PH_DET")
                for item in context["pattern_recall_signals"]
            )
        )

    def test_reasoning_inference_source_is_rejected(self):
        evidence = radiation_evidence()
        evidence.observations[0].source = "reasoning_inference"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [radiation_hypothesis()]},
            evidence,
        )
        rejected = context["pattern_verification_results"][0]
        self.assertEqual(rejected["verification_status"], "rejected")
        self.assertIn("reasoning_inference_source", rejected["rejection_reasons"])
        self.assertEqual(context["unverified_pattern_leakage_count"], 0)

    def test_radiation_pattern_recall_adds_candidate_without_evidence_leakage(self):
        evidence = radiation_evidence()
        llm = {"clinical_pattern_hypotheses": [radiation_hypothesis()]}
        decision = self.engine.decide(llm, [], evidence)
        top20 = [item.diagnosis for item in decision.candidates[:20]]
        self.assertIn(RAD_PNEUMONITIS, top20)
        candidate = next(item for item in decision.candidates if item.diagnosis == RAD_PNEUMONITIS)
        self.assertTrue(
            any(
                source.get("source") == "llm_pattern_hypothesis"
                and source.get("metadata", {}).get("judge_evidence_weight") == 0.0
                and source.get("metadata", {}).get("eligibility_evidence_weight") == 0.0
                and source.get("metadata", {}).get("active_gap_write_permission") == "none"
                for source in candidate.candidate_sources
            )
        )
        self.assertNotIn("post_thoracic_radiotherapy_lung_injury_pattern", candidate.matched_evidence)
        self.assertEqual(decision.unverified_pattern_leakage_count, 0)
        self.assertEqual(decision.pattern_generated_active_gaps, 0)
        self.assertLessEqual(decision.pattern_expansion_round_count, 1)
        self.assertTrue(decision.pattern_gap_suggestions)

    def test_generic_radiotherapy_history_cannot_be_protected_recall(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "history_of_radiotherapy",
                    "patient_reported_observation",
                    confidence=0.9,
                    information_value=0.5,
                ),
                Observation(
                    "post_radiotherapy_time_window",
                    "disease_agnostic_deterministic_relation",
                    confidence=0.9,
                    information_value=0.86,
                ),
                Observation(
                    "ground_glass_opacity",
                    "imaging_result",
                    confidence=0.92,
                    information_value=0.92,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    confidence=0.88,
                    information_value=0.62,
                ),
            ]
        )
        payload = radiation_hypothesis()
        payload["evidence_bindings"][0]["evidence_id"] = "history_of_radiotherapy"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
        )
        result = context["pattern_verification_results"][0]
        self.assertNotEqual(result["verification_status"], "verified")
        self.assertEqual(context["pattern_protected_candidate_recall"], [])

    def test_pattern_recall_merges_existing_mitral_regurgitation_entity(self):
        evidence = EvidenceBundle(
            [
                Observation("cardiac_murmur", "clinician_observed_finding", confidence=0.96, information_value=0.96),
                Observation("left_heart_enlargement", "imaging_result", confidence=0.96, information_value=0.96),
                Observation("dyspnea", "patient_reported_observation", confidence=0.92, information_value=0.9),
            ]
        )
        llm = {
            "clinical_pattern_hypotheses": [
                {
                    "pattern_hypothesis_id": "PH_MR_001",
                    "pattern_name": "left_sided_valvular_regurgitation_pattern",
                    "pattern_type": "mechanism_multievidence",
                    "evidence_bindings": [
                        {
                            "evidence_id": "cardiac_murmur",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "organ_manifestation",
                        },
                        {
                            "evidence_id": "left_heart_enlargement",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "imaging_or_objective_finding",
                        },
                        {
                            "evidence_id": "dyspnea",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "support",
                        },
                    ],
                    "relations": [
                        {
                            "type": "anatomical_consistency",
                            "from": "cardiac_murmur",
                            "to": "left_heart_enlargement",
                        }
                    ],
                    "suggested_diseases": [
                        {"name": MITRAL_REGURGITATION, "canonical_id": "D100012"}
                    ],
                    "missing_evidence_requests": [
                        {"target_evidence": "echo_regurgitant_jet"}
                    ],
                }
            ]
        }
        decision = self.engine.decide(llm, [], evidence)
        candidate = next(item for item in decision.candidates if item.entity_id == "D100012")
        self.assertEqual(candidate.diagnosis, MITRAL_REGURGITATION)
        self.assertTrue(
            any(source.get("source") == "llm_pattern_hypothesis" for source in candidate.candidate_sources)
        )
        self.assertEqual(decision.pattern_generated_active_gaps, 0)

    def test_thinking_structured_pattern_can_drive_recall(self):
        evidence = radiation_evidence()
        thinking = {
            "differential_diagnosis": [
                {"diagnosis": "\u652f\u6c14\u7ba1\u80ba\u708e", "likelihood": 0.5}
            ],
            "clinical_pattern_proposals": [radiation_hypothesis()],
            "key_unknowns": ["infection_exclusion"],
            "is_sufficient": False,
        }
        snapshot = ThinkingSnapshot.from_thinking(
            thinking,
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertEqual(context["thinking_snapshot_count"], 1)
        self.assertEqual(len(context["pattern_hypotheses"]), 1)
        self.assertTrue(context["pattern_recall_signals"])
        audit = context["pattern_recall_audit"]
        self.assertEqual(audit["proposal_count"], 1)
        self.assertEqual(audit["verification_statuses"]["verified"], 1)
        self.assertIn("D100058", audit["signal_entity_ids"])
        decision = self.engine.decide({}, [], evidence, pattern_recall_context=context)
        top20 = [item.diagnosis for item in decision.candidates[:20]]
        self.assertIn(RAD_PNEUMONITIS, top20)
        self.assertEqual(decision.pattern_recall_audit["proposal_count"], 1)
        self.assertTrue(
            any(
                item.get("entity_id") == "D100058"
                and item.get("admitted_to_controlled_pool")
                for item in decision.pattern_candidate_admissions
            )
        )

    def test_thinking_disease_name_only_does_not_strong_recall_from_thinking(self):
        evidence = radiation_evidence()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "differential_diagnosis": [
                    {"diagnosis": RAD_PNEUMONITIS, "likelihood": 0.8}
                ],
                "action_reasoning": "\u8003\u8651\u653e\u5c04\u6027\u80ba\u708e",
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertTrue(context["pattern_hypotheses"])
        self.assertTrue(
            all(
                item["generator_source"] == "deterministic_relation"
                for item in context["pattern_hypotheses"]
            )
        )
        self.assertTrue(context["pattern_recall_signals"])
        self.assertEqual(
            context["pattern_recall_audit"]["compiler_audit"]["sources"]["reasoning_adapter"]["generated"],
            0,
        )

    def test_thinking_differential_with_refs_is_not_protected_without_relations(self):
        evidence = radiation_evidence()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "differential_diagnosis": [
                    {
                        "diagnosis": RAD_PNEUMONITIS,
                        "likelihood": 0.8,
                        "supporting_evidence_refs": [
                            "thoracic_radiotherapy",
                            "ground_glass_opacity",
                        ],
                    }
                ],
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertTrue(context["pattern_hypotheses"])
        self.assertTrue(context["pattern_protected_candidate_recall"])
        self.assertTrue(
            all(
                item["pattern_hypothesis_id"].startswith("PH_DET")
                for item in context["pattern_protected_candidate_recall"]
            )
        )
        self.assertEqual(
            context["pattern_recall_audit"]["compiler_audit"]["sources"]["reasoning_adapter"]["generated"],
            0,
        )

    def test_thinking_and_diagnosis_draft_duplicate_pattern_is_deduped(self):
        evidence = radiation_evidence()
        payload = radiation_hypothesis()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "clinical_pattern_proposals": [payload],
                "differential_diagnosis": [],
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertEqual(len(context["pattern_hypotheses"]), 1)

    def test_family_relation_can_recall_entity_without_disease_name(self):
        evidence = EvidenceBundle(
            [
                Observation("cyanosis", "patient_reported_observation", confidence=0.9, information_value=0.86),
                Observation("hypoxemia", "laboratory_result", confidence=0.9, information_value=0.9),
                Observation(
                    "pulmonary_vascular_abnormality",
                    "imaging_result",
                    confidence=0.86,
                    information_value=0.88,
                ),
            ]
        )
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "clinical_pattern_proposals": [
                    {
                        "pattern_hypothesis_id": "PH_PVASC_001",
                        "pattern_name": "pulmonary_vascular_shunt_or_malformation_pattern",
                        "pattern_type": "vascular_shunt",
                        "suggested_family": "pulmonary_vascular_shunt",
                        "evidence_bindings": [
                            {
                                "evidence_id": "cyanosis",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "organ_manifestation",
                            },
                            {
                                "evidence_id": "hypoxemia",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "imaging_or_objective_finding",
                            },
                            {
                                "evidence_id": "pulmonary_vascular_abnormality",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "support",
                            },
                        ],
                        "relations": [
                            {
                                "type": "anatomical_consistency",
                                "from_evidence_ref": "pulmonary_vascular_abnormality",
                                "to_evidence_ref": "hypoxemia",
                            }
                        ],
                        "suggested_diseases": [],
                    }
                ]
            },
            case_id="Patient_03998",
            patient_id="Patient_03998",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03998",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        signals = context["pattern_recall_signals"]
        self.assertTrue(any(item["entity_id"] == PAVM_ENTITY_ID for item in signals))
        self.assertEqual(context["pattern_protected_candidate_recall"], [])


if __name__ == "__main__":
    unittest.main()
