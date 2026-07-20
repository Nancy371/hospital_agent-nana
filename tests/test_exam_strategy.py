import unittest

from agent.exam_strategy import ExamStrategyAgent
from agent.knowledge import KnowledgeBase


class ExamInformationGainTests(unittest.TestCase):
    def setUp(self):
        self.strategy = ExamStrategyAgent(KnowledgeBase("data/ref_data"), max_new_items=6)

    def test_top_three_candidates_drive_exam_ranking(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["上腹痛", "恶心", "呕吐"]},
            candidate_diseases=["胃炎", "胆囊炎", "胰腺炎"],
            proposed_items=["超声心动图", "腹部B超", "血常规"],
            existing_results={},
        )
        self.assertLessEqual(len(result["items"]), 6)
        self.assertTrue(result["information_gain"])
        self.assertNotEqual(result["items"][0], "超声心动图")
        self.assertNotIn("心导管检查", result["items"])

    def test_completed_exam_is_not_reordered(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["尿频", "尿急", "尿痛"]},
            candidate_diseases=["泌尿系感染", "肾结石"],
            proposed_items=["尿常规", "腹部B超"],
            existing_results={"尿常规": {"status": "normal"}},
        )
        self.assertNotIn("尿常规", result["items"])

    def test_electrolyte_crisis_path_prioritizes_metabolic_exams(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["腹泻", "手足抽筋", "心悸", "意识模糊"],
                "chief_complaint": "腹泻后手足抽筋和心悸",
            },
            candidate_diseases=["心律失常", "低镁血症"],
            proposed_items=["经食管超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("综合代谢面板（CMP）", result["items"])
        self.assertIn("24小时尿电解质检测", result["items"])
        self.assertIn("镁负荷试验", result["items"])
        self.assertIn("心电图（ECG）", result["items"])
        self.assertIn("综合代谢面板（CMP）", result["strong_verification_items"])
        self.assertIn("电解质", result["items"])
        self.assertIn("肾功能", result["items"])
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])
        self.assertNotIn("心脏MRI（CMR）", result["items"])

    def test_pulmonary_renal_path_suppresses_advanced_cardiac_package(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["咳血痰", "尿色加深", "脚踝水肿", "气短"],
                "chief_complaint": "咳血痰、气短、尿色加深",
            },
            candidate_diseases=["冠心病", "显微镜下多血管炎"],
            proposed_items=["心肌酶谱", "经食管超声心动图", "三维超声心动图"],
            existing_results={},
        )
        self.assertIn("尿液分析（UA）", result["items"])
        self.assertIn("肾功能", result["items"])
        self.assertIn("胸部CT扫描（Chest CT）", result["items"])
        self.assertIn("抗核抗体", result["items"])
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertIn("C反应蛋白（CRP）", result["items"])
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])
        self.assertNotIn("三维超声心动图（3D Echo）", result["items"])

    def test_metabolic_bone_path_uses_rickets_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["腿痛", "跛行", "运动耐力下降"]},
            candidate_diseases=["维生素D缺乏性佝偻病"],
            proposed_items=["超声心动图", "胸部X线"],
            existing_results={},
        )
        self.assertEqual(
            result["items"][:6],
            [
                "维生素D检测",
                "血清电解质",
                "甲状旁腺激素检测（PTH）",
                "肝功能检查（LFTs）",
                "骨转换标志物（BTMs）",
                "X线检查",
            ],
        )
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])

    def test_aspiration_path_uses_pulmonary_infection_and_atelectasis_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=12)
        result = strategy.recommend(
            collected_info={"symptoms": ["呛咳", "咳嗽", "发热", "呼吸困难"]},
            candidate_diseases=["肺不张", "支气管肺炎"],
            proposed_items=["超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("脉搏血氧饱和度监测（SpO2）", result["items"])
        self.assertIn("动脉血气（ABG）", result["items"])
        self.assertIn("胸部X线检查（CXR）", result["items"])
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertIn("支气管镜检查", result["items"])
        self.assertIn("抗菌药物敏感性试验（AST）", result["items"])
        self.assertNotIn("心脏MRI（CMR）", result["items"])

    def test_advanced_cardiac_exams_remain_when_structural_signal_is_explicit(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["活动后气短", "下肢水肿"],
                "physical_signs": "心尖部收缩期杂音，考虑二尖瓣反流",
            },
            candidate_diseases=["二尖瓣反流", "心力衰竭"],
            proposed_items=["经食管超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("经食管超声心动图（TEE）", result["items"])
        self.assertIn("心脏MRI（CMR）", result["items"])

    def test_corrective_exam_gate_blocks_advanced_cardiac_without_signal(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        items = strategy.prepare_order_items(
            ["经食管超声心动图", "心脏MRI", "三维超声心动图", "电解质"],
            collected_info={"symptoms": ["腿痛", "跛行", "运动耐量下降"]},
            candidate_diseases=["维生素D缺乏性佝偻病"],
            existing_results={},
            max_items=8,
        )
        self.assertNotIn("经食管超声心动图（TEE）", items)
        self.assertNotIn("心脏MRI（CMR）", items)
        self.assertNotIn("三维超声心动图（3D Echo）", items)
        self.assertIn("维生素D检测", items)

    def test_generic_electrolyte_does_not_globally_become_cmp(self):
        normalized, invalid = self.strategy.knowledge.normalize_examinations(["电解质"])
        self.assertEqual(normalized, ["电解质"])
        self.assertEqual(invalid, [])

    def test_new_backend_standard_aliases_normalize(self):
        normalized, invalid = self.strategy.knowledge.normalize_examinations(
            ["镁负荷", "24小时尿电解质", "PTH", "BTMs"]
        )
        self.assertEqual(
            normalized,
            [
                "镁负荷试验",
                "24小时尿电解质检测",
                "甲状旁腺激素检测（PTH）",
                "骨转换标志物（BTMs）",
            ],
        )
        self.assertEqual(invalid, [])


if __name__ == "__main__":
    unittest.main()
