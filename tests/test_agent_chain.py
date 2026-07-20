import copy
import time
import unittest
from contextlib import nullcontext

import yaml

from agent.agent import MyDoctorAgent
from agent.diagnosis_critic import CriticDecision


class FakeActions:
    def __init__(self, events):
        self.events = events

    async def ask_patient(self, patient_id, input_data):
        self.events.append("ask_patient")
        return "手足抽筋并伴心悸。"

    async def order_examination(self, patient_id, items, reason=""):
        self.events.append("order_examination")
        return {
            "results": {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            }
        }

    async def prescribe_treatment(self, patient_id, diagnosis, treatment_plan, reasoning):
        self.events.append("prescribe_treatment")
        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "finished": True,
        }


class AgentChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_actions_follow_evidence_first_submission_order(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with nullcontext("tests/_runtime_chain") as tmp:
            config = copy.deepcopy(config)
            config["memory"]["json_path"] = f"{tmp}/memory.json"
            config["memory"]["md_path"] = f"{tmp}/memory.md"
            config["memory"]["diagnostic_replay_path"] = f"{tmp}/replay.jsonl"
            config["policy_store_path"] = f"{tmp}/policies.json"
            config["self_improve_enabled"] = False
            events = []
            agent = MyDoctorAgent(config)
            agent.actions = FakeActions(events)
            agent._case_started_at = time.monotonic()
            agent._case_deadline = agent._case_started_at + 235

            answer = await agent.actions.ask_patient("case-chain", {"question": "症状？"})
            exam_response = await agent.actions.order_examination(
                "case-chain", ["电解质"], "确认电解质异常"
            )
            exam_results = exam_response["results"]

            async def fake_diagnosis(_messages):
                return {
                    "diagnosis": ["低镁血症"],
                    "treatment_plan": "结合肾功能补镁并复查电解质。",
                    "reasoning": "症状与低血镁一致。",
                }

            original_search = agent.memory_manager.search_rag
            original_decide = agent.diagnosis_engine.decide
            original_review = agent.diagnosis_critic.review
            original_safety = agent.treatment_safety.review

            def traced_search(*args, **kwargs):
                events.append("rag")
                return original_search(*args, **kwargs)

            def traced_decide(*args, **kwargs):
                events.append("evidence_score")
                return original_decide(*args, **kwargs)

            async def traced_critic(decision, evidence, remaining_seconds, allow_llm=True):
                events.append("critic")
                return CriticDecision(
                    selected_diagnoses=list(decision.final_diagnoses),
                    confidence=decision.confidence,
                    reason="测试确定性审查",
                )

            def traced_safety(*args, **kwargs):
                events.append("treatment_safety")
                return original_safety(*args, **kwargs)

            agent._llm_generate_diagnosis = fake_diagnosis
            agent.memory_manager.search_rag = traced_search
            agent.diagnosis_engine.decide = traced_decide
            agent.diagnosis_critic.review = traced_critic
            agent.treatment_safety.review = traced_safety

            result = await agent._prescribe(
                patient_id="case-chain",
                collected_info={"symptoms": [answer]},
                exam_results=exam_results,
                chat_history=[
                    {"from": "doctor", "text": "症状？"},
                    {"from": "patient", "text": answer},
                ],
            )

            expected_order = [
                "ask_patient",
                "order_examination",
                "rag",
                "evidence_score",
                "rag",
                "evidence_score",
                "critic",
                "treatment_safety",
                "prescribe_treatment",
            ]
            cursor = 0
            for event in events:
                if cursor < len(expected_order) and event == expected_order[cursor]:
                    cursor += 1
            self.assertEqual(cursor, len(expected_order), events)
            self.assertEqual(result["diagnosis"][0], "低镁血症")
            self.assertTrue(result["finished"])


class EvidenceGapExamRecommendationTests(unittest.TestCase):
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

    def test_low_magnesium_gap_recommends_metabolic_confirmation(self):
        agent = self.make_agent()
        info = {"symptoms": ["腹泻", "手足抽筋", "心悸", "意识模糊"]}
        evidence = agent.clinical_normalizer.normalize(info, {})
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": "心律失常", "confidence": 0.9},
                    {"name": "低镁血症", "confidence": 0.82},
                ]
            },
            [],
            evidence,
        )
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertLessEqual(len(exams), 4)
        self.assertIn("综合代谢面板（CMP）", exams)
        self.assertIn("24小时尿电解质检测", exams)

    def test_pulmonary_renal_gap_recommends_vasculitis_workup(self):
        agent = self.make_agent()
        info = {"symptoms": ["咳血痰", "尿色加深", "脚踝水肿", "气短"]}
        evidence = agent.clinical_normalizer.normalize(info, {})
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": "冠心病", "confidence": 0.8},
                    {"name": "显微镜下多血管炎", "confidence": 0.76},
                ]
            },
            [],
            evidence,
        )
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertLessEqual(len(exams), 4)
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", exams)
        self.assertIn("MPO-ANCA", exams)
        self.assertIn("尿液分析（UA）", exams)
        self.assertIn("肾功能", exams)

    def test_final_name_filter_drops_generic_pneumonia_when_child_selected(self):
        agent = self.make_agent()
        filtered = agent._remove_suppressed_diagnosis_names(["支气管肺炎", "肺不张", "肺炎"])
        self.assertEqual(filtered, ["支气管肺炎", "肺不张"])

    def test_refilter_result_writes_back_suppressed_diagnosis_changes(self):
        agent = self.make_agent()
        info = {"symptoms": ["呛咳", "发热", "咳嗽", "呼吸困难"]}
        exams = {
            "胸部CT扫描（Chest CT）": {
                "status": "abnormal",
                "result": {"结论": "右下叶实变并见空气支气管征，肺不张"},
            },
            "支气管镜检查": {
                "status": "abnormal",
                "result": {"结论": "支气管内见脓性分泌物"},
            },
        }
        evidence = agent.clinical_normalizer.normalize(info, exams)
        decision = agent.diagnosis_engine.decide(
            {"diagnosis_candidates": ["支气管肺炎", "肺不张", "肺炎"]},
            [],
            evidence,
        )
        fixed = agent._refilter_diagnosis_result(
            {"diagnosis": ["支气管肺炎", "肺不张", "肺炎"], "reasoning": ""},
            decision,
            evidence,
        )
        self.assertNotIn("肺炎", fixed["diagnosis"])
        self.assertTrue({"支气管肺炎", "肺不张"} <= set(fixed["diagnosis"]))

    def test_critic_selection_keeps_weak_lung_cancer_as_differential_only(self):
        agent = self.make_agent()
        info = {"symptoms": ["咳血痰", "尿色加深", "全身酸痛", "脚踝水肿", "呼吸困难"]}
        exams = {
            "尿液分析（UA）": {
                "status": "abnormal",
                "result": {
                    "尿红细胞": "50 个/HPF［参考值：0-3］",
                    "尿蛋白": "尿蛋白阳性",
                },
            },
            "抗中性粒细胞胞质抗体（ANCA）谱": {
                "status": "abnormal",
                "result": {"MPO-ANCA": "MPO-ANCA阳性"},
            },
            "胸部CT扫描（Chest CT）": {
                "status": "abnormal",
                "result": {"结论": "弥漫性肺泡出血"},
            },
        }
        evidence = agent.clinical_normalizer.normalize(info, exams)
        decision = agent.diagnosis_engine.decide(
            {"diagnosis_candidates": ["显微镜下多血管炎", "肺癌"]},
            [],
            evidence,
        )

        agent._apply_critic_selection(
            decision,
            ["显微镜下多血管炎", "肺癌"],
            "显微镜下多血管炎最能解释肺肾综合征，肺癌需鉴别。",
        )
        fixed = agent.diagnosis_engine.apply_to_result(
            {"diagnosis": list(decision.final_diagnoses), "reasoning": ""},
            decision,
            evidence,
        )

        self.assertEqual(fixed["diagnosis"], ["显微镜下多血管炎"])
        differential_only = fixed["_diagnosis_decision"]["differential_only_diagnoses"]
        self.assertIn("肺癌", [item["diagnosis"] for item in differential_only])
        self.assertIn("仅鉴别", fixed["reasoning"])


if __name__ == "__main__":
    unittest.main()
