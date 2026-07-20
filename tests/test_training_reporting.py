import copy
import time
import unittest
from unittest.mock import Mock

import yaml

from agent.agent import MyDoctorAgent
from agent.knowledge import KnowledgeBase
from hospital_agent.base import summarize_training_results


class TrainingReportingTests(unittest.TestCase):
    def test_batch_summary_separates_evaluation_and_reflection_errors(self):
        rows = [
            {
                "status": "evaluated",
                "metrics": {
                    "diagnosis_accuracy": 1.0,
                    "examination_precision": 0.8,
                    "treatment_overall_score": 0.7,
                    "treatment_safety": 1.0,
                    "candidate_recall_at_5": True,
                },
                "audit": {
                    "elapsed_seconds": 100,
                    "timed_out": False,
                    "critic_issues": ["narrow_margin:0.1"],
                    "critic_llm_used": True,
                },
                "evaluation_error": "",
                "reflection_error": "memory write failed",
            },
            {
                "status": "evaluation_failed",
                "metrics": {},
                "audit": {
                    "elapsed_seconds": 235,
                    "timed_out": True,
                    "critic_issues": [],
                    "critic_llm_used": False,
                },
                "evaluation_error": "backend 503",
                "reflection_error": "",
            },
        ]
        summary = summarize_training_results(rows)
        self.assertEqual(summary["diagnosis_accuracy"], 1.0)
        self.assertEqual(summary["candidate_recall_at_5"], 1.0)
        self.assertEqual(summary["critic_issue_rate"], 0.5)
        self.assertEqual(summary["critic_llm_rate"], 0.5)
        self.assertEqual(summary["timeout_cases"], 1)
        self.assertEqual(summary["backend_error_cases"], 1)
        self.assertEqual(summary["reflection_error_cases"], 1)

    def test_exam_alias_feedback_stays_pending_when_auto_promotion_is_disabled(self):
        knowledge = KnowledgeBase(
            ref_dir="data/ref_data",
            allow_auto_alias_promotion=False,
        )
        knowledge._read_json_file = Mock(return_value={"candidates": []})
        knowledge._write_json_file = Mock()
        knowledge._infer_exam_alias_standard = Mock(return_value="尿液分析（UA）")
        knowledge._promote_pending_exam_aliases = Mock(return_value=1)
        report = {
            "diagnosisAccuracy": 1.0,
            "examinationPrecision": 1.0,
            "treatmentOverallScore": 1.0,
            "examinationDetail": {
                "expected": ["尿液分析（UA）"],
                "ordered": ["UA-new-alias"],
            },
        }
        result = knowledge.record_exam_alias_feedback(
            "case-alias", report, ["UA-new-alias"]
        )
        self.assertEqual(result, {"pending": 1, "promoted": 0})
        knowledge._promote_pending_exam_aliases.assert_not_called()

    def test_training_record_contains_replay_metrics_without_internal_payload(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = copy.deepcopy(yaml.safe_load(handle))
        config["self_improve_enabled"] = False
        config["memory"]["json_path"] = "tests/_training_report_memory.json"
        config["memory"]["md_path"] = "tests/_training_report_memory.md"
        agent = MyDoctorAgent(config)
        agent._last_diagnosis_audit = {
            "diagnosis_decision": {
                "candidates": [
                    {"diagnosis": "低镁血症"},
                    {"diagnosis": "心律失常"},
                ]
            },
            "critic": {"issues": [], "llm_used": False},
            "elapsed_seconds": 88.5,
            "timed_out": False,
        }
        report = {
            "diagnosisAccuracy": 1.0,
            "examinationPrecision": 0.9,
            "treatmentOverallScore": 0.8,
            "treatmentSafety": 1.0,
            "diagnosisDetail": {
                "expected": ["低镁血症"],
                "submitted": ["低镁血症"],
            },
        }
        result = agent._build_training_result(
            "case-report",
            {
                "patient_id": "case-report",
                "diagnosis": ["低镁血症"],
                "treatment_plan": "补镁并监测。",
                "reasoning": "低血镁。",
                "_private": "must not leak",
            },
            report,
        )
        self.assertTrue(result["metrics"]["candidate_recall_at_5"])
        self.assertEqual(result["audit"]["elapsed_seconds"], 88.5)
        self.assertNotIn("_private", result["final_result"])


class TrainingBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_clinical_deadline_preserves_training_post_submit_budget(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = copy.deepcopy(yaml.safe_load(handle))
        config["self_improve_enabled"] = False
        config["execution"]["case_timeout_seconds"] = 10
        config["execution"]["fallback_reserve_seconds"] = 2
        config["memory"]["json_path"] = "tests/_training_budget_memory.json"
        config["memory"]["md_path"] = "tests/_training_budget_memory.md"
        agent = MyDoctorAgent(config)

        async def completed_pipeline(_patient_id):
            return {"diagnosis": ["低镁血症"], "finished": True}

        agent._execute_with_planner = completed_pipeline
        before = time.monotonic()
        await agent._run_case_pipeline(
            "case-budget",
            post_submit_reserve_seconds=4,
        )
        self.assertGreaterEqual(agent._case_deadline, before + 9.9)
        self.assertAlmostEqual(
            agent._case_deadline - agent._case_clinical_deadline,
            4.0,
            places=2,
        )
        await agent._cleanup()


if __name__ == "__main__":
    unittest.main()
