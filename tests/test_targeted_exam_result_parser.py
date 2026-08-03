import copy
import unittest

import yaml

from agent.agent import MyDoctorAgent
from agent.clinical_evidence import EvidenceBundle, Observation
from agent.evidence_pattern_compiler import EvidencePatternCompiler
from agent.targeted_exam_result_parser import (
    ExamResultIntentBinding,
    TargetedExamResultParser,
)


class TargetedExamResultParserTests(unittest.TestCase):
    def test_enhanced_ct_bound_to_pavm_gap_recovers_vascular_anchor(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-1",
            order_id="O-PAVM-1",
            requested_exam="胸部增强CT",
            resolved_exam="胸部增强CT",
            actual_result_exam="增强胸部CT扫描（Chest CECT）",
            target_gap_ids=["G-PAVF-01"],
            target_claims=["enhanced_ct_vascular_malformation"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "结论": "右下叶强化迂曲血管性病变，可见供血肺动脉及早期引流肺静脉，考虑肺动静脉畸形。"
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.status, "positive")
        self.assertEqual(parsed.gap_closure_assessment, "positive_closed")
        self.assertIn("feeding_pulmonary_artery_present", findings)
        self.assertIn("draining_pulmonary_vein_present", findings)
        self.assertIn("enhanced_ct_vascular_malformation", findings)
        self.assertTrue(all(item.target_gap_ids == ["G-PAVF-01"] for item in parsed.observations))

        compiler = EvidencePatternCompiler(ref_dir="data/ref_data")
        derived = compiler.compile(
            [],
            EvidenceBundle(
                [
                    Observation("hemoptysis", "问诊"),
                    *parsed.observations,
                ]
            ),
        )
        self.assertIn("pulmonary_av_fistula_pattern", {item.finding for item in derived})

    def test_standard_echo_normal_does_not_negate_bubble_echo_gap(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-2",
            order_id="O-PAVM-2",
            requested_exam="右心声学造影",
            resolved_exam="超声心动图右心声学造影",
            actual_result_exam="超声心动图",
            target_gap_ids=["G-PAVF-02"],
            target_claims=["bubble_echo_right_to_left_shunt"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {"status": "normal", "result": {"结论": "心脏结构未见明显异常。"}},
            binding,
        )

        self.assertIn(parsed.status, {"unresolved", "inconclusive"})
        self.assertNotIn(
            "bubble_echo_right_to_left_shunt",
            {item.finding for item in parsed.observations},
        )
        self.assertNotEqual(parsed.gap_closure_assessment, "negative_closed")

    def test_plain_ct_cannot_create_pavm_confirmatory_pattern(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-PLAIN",
            order_id="O-PAVM-PLAIN",
            requested_exam="肺动脉CTA",
            resolved_exam="肺动脉CTA",
            actual_result_exam="胸部CT扫描（Chest CT）",
            target_gap_ids=["G-PAVF-PLAIN"],
            target_claims=["pulmonary_cta_positive"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "结论": "右下肺结节样血管影增粗，可疑供血动脉及引流静脉，建议增强检查。"
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.gap_closure_assessment, "partial")
        self.assertIn("vascular_pulmonary_nodule_suspected", findings)
        self.assertNotIn("feeding_pulmonary_artery_present", findings)
        self.assertNotIn("draining_pulmonary_vein_present", findings)

        compiler = EvidencePatternCompiler(ref_dir="data/ref_data")
        derived = compiler.compile(
            [],
            EvidenceBundle(
                [
                    Observation("hemoptysis", "问诊"),
                    Observation("cyanosis", "体格检查"),
                    *parsed.observations,
                ]
            ),
        )
        self.assertNotIn("pulmonary_av_fistula_pattern", {item.finding for item in derived})

    def test_effective_cta_negative_can_close_gap_negative(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-3",
            order_id="O-PAVM-3",
            requested_exam="肺动脉CTA",
            resolved_exam="肺动脉CTA",
            actual_result_exam="肺动脉CTA",
            target_gap_ids=["G-PAVF-03"],
            target_claims=["pulmonary_cta_positive"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {"status": "normal", "result": {"结论": "未见肺动静脉异常交通，未见肺血管畸形。"}},
            binding,
        )

        self.assertEqual(parsed.status, "negative")
        self.assertEqual(parsed.gap_closure_assessment, "negative_closed")
        self.assertTrue(any(item.polarity == "negative" for item in parsed.observations))


class AgentTargetedExamRecoveryTests(unittest.TestCase):
    def make_agent(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config = copy.deepcopy(config)
        config["memory"]["json_path"] = "tests/_runtime_chain/memory.json"
        config["memory"]["md_path"] = "tests/_runtime_chain/memory.md"
        config["memory"]["diagnostic_replay_path"] = "tests/_runtime_chain/replay.jsonl"
        config["policy_store_path"] = "tests/_runtime_chain/policies.json"
        config["self_improve_enabled"] = False
        return MyDoctorAgent(config)

    def test_agent_binds_returned_cect_name_to_original_pavm_gap(self):
        agent = self.make_agent()
        strategy = {
            "exam_authorization_details": [
                {
                    "exam": "胸部增强CT",
                    "requested_exam": "胸部增强CT",
                    "resolved_exam": "胸部增强CT",
                    "exam_source": "deferred_gap_closure_exam",
                    "target_gaps": ["G-PAVF-01"],
                    "target_findings": ["enhanced_ct_vascular_malformation"],
                    "target_candidates": ["肺动静脉瘘"],
                    "source_gap_value": 0.91,
                }
            ]
        }
        new_results = {
            "增强胸部CT扫描（Chest CECT）": {
                "status": "abnormal",
                "result": {
                    "结论": "右下肺异常血管团，见供血肺动脉及早期引流肺静脉。"
                },
            }
        }

        agent._record_targeted_exam_result_recovery(
            patient_id="Patient_03998",
            stage="unit_test",
            ordered_items=["胸部增强CT"],
            new_results=new_results,
            strategy=strategy,
        )

        self.assertTrue(agent._exam_result_intent_bindings)
        binding = agent._exam_result_intent_bindings[0]
        self.assertEqual(binding["actual_result_exam"], "增强胸部CT扫描（Chest CECT）")
        findings = {item.finding for item in agent._targeted_exam_observations}
        self.assertIn("enhanced_ct_vascular_malformation", findings)
        evidence = agent._normalize_with_exam_recovery(
            {"symptoms": ["咯血", "低氧"]},
            new_results,
        )
        self.assertIn(
            "enhanced_ct_vascular_malformation",
            set(evidence.findings("positive")),
        )
        self.assertIn(
            "pulmonary_av_fistula_pattern",
            set(evidence.findings("positive")),
        )


if __name__ == "__main__":
    unittest.main()
