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

    def test_differential_driven_plan_uses_judge_discriminating_exams(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["皮疹", "关节痛", "乏力"]},
            candidate_diseases=["雅司病", "湿疹", "白血病"],
            proposed_items=["超声心动图", "心电图"],
            existing_results={},
            judge_decision={
                "primary": "雅司病",
                "differential_candidates": ["雅司病", "湿疹", "白血病"],
                "discriminating_exams": [
                    "全血细胞计数（CBC）",
                    "外周血涂片",
                    "梅毒螺旋体血清学试验",
                    "体格检查",
                    "组织病理学检查",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertLessEqual(len(result["items"]), 4)
        self.assertTrue(result["exam_authorization_details"])
        self.assertTrue(
            all(
                item["exam_source"] == "judge_discriminating_exam"
                for item in result["exam_authorization_details"]
            )
        )
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertIn("梅毒血清学检查", result["items"])
        self.assertNotIn("超声心动图", result["items"])
        self.assertNotIn("心电图（ECG）", result["items"])

    def test_tb_differential_plan_uses_shared_discriminating_exams(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["咳嗽", "咯血", "低热", "消瘦"]},
            candidate_diseases=["肺结核", "肺炎", "肺癌"],
            proposed_items=["泌尿道超声", "尿动力学"],
            existing_results={},
            judge_decision={
                "primary": "肺结核",
                "differential_candidates": ["肺结核", "肺炎", "肺癌"],
                "discriminating_exams": [
                    "胸部CT扫描（Chest CT）",
                    "痰培养",
                    "抗酸杆菌染色（AFB）",
                    "Xpert MTB/RIF",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertEqual(
            result["items"],
            [
                "胸部CT扫描（Chest CT）",
                "痰培养",
                "抗酸杆菌染色（AFB）",
                "核酸扩增检测（NAAT）",
            ],
        )
        self.assertNotIn("泌尿道超声", result["items"])
        self.assertNotIn("尿动力学检查（UDS）", result["items"])

    def test_deferred_judge_state_takes_priority_over_strict_primary(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["皮肤瘙痒", "低热", "腹股沟淋巴结肿大"]},
            candidate_diseases=["水痘", "雅司病"],
            proposed_items=["体格检查", "血清学抗体检测", "胸部X线"],
            existing_results={},
            judge_decision={
                "primary": "水痘",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "provisional_primary": "水痘",
                "differential_candidates": ["水痘", "雅司病", "白血病"],
                "discriminating_exams": [
                    "全血细胞计数（CBC）",
                    "外周血涂片",
                    "梅毒血清学检查",
                    "体格检查",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertFalse(result["strict_diagnosis_driven"])
        self.assertTrue(
            all(
                item["allowed_reason"] == "needs_discriminating_exams"
                for item in result["exam_authorization_details"]
            )
        )
        self.assertEqual(result["items"][0], "全血细胞计数（CBC）")
        self.assertIn("梅毒血清学检查", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"])

    def test_judge_needing_discrimination_blocks_legacy_package_without_targets(self):
        result = self.strategy.recommend(
            collected_info={
                "symptoms": [
                    "\u70ed\u5e26\u5730\u533a\u751f\u6d3b\u540e\u6df1\u90e8\u6e83\u75a1\u7ed3\u75c2",
                    "\u5c40\u90e8\u9aa8\u819c\u708e",
                ]
            },
            candidate_diseases=["\u96c5\u53f8\u75c5", "\u6e7f\u75b9"],
            proposed_items=[
                "\u8179\u90e8\u8d85\u58f0",
                "\u809d\u529f\u80fd\u68c0\u67e5\uff08LFTs\uff09",
                "\u7efc\u5408\u4ee3\u8c22\u9762\u677f\uff08CMP\uff09",
                "\u80be\u529f\u80fd",
            ],
            existing_results={},
            judge_decision={
                "primary": "\u96c5\u53f8\u75c5",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "provisional_primary": "\u96c5\u53f8\u75c5",
                "differential_candidates": ["\u96c5\u53f8\u75c5", "\u6e7f\u75b9"],
                "discriminating_exams": [],
            },
        )
        self.assertEqual(result["items"], [])
        self.assertFalse(result["strict_diagnosis_driven"])
        self.assertFalse(result["differential_driven"])
        self.assertIn("\u8179\u90e8\u8d85\u58f0", result["blocked_items"])
        self.assertIn("\u809d\u529f\u80fd\u68c0\u67e5\uff08LFTs\uff09", result["blocked_items"])
        self.assertIn("\u7efc\u5408\u4ee3\u8c22\u9762\u677f\uff08CMP\uff09", result["blocked_items"])
        self.assertIn("\u80be\u529f\u80fd\u68c0\u67e5\uff08RFTs\uff09", result["blocked_items"])

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
        self.assertNotIn("电解质", result["items"])
        self.assertNotIn("肾功能检查（RFTs）", result["items"])
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
        self.assertIn("肾功能检查（RFTs）", result["items"])
        self.assertIn("胸部CT扫描（Chest CT）", result["items"])
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", result["items"])
        self.assertIn("MPO-ANCA", result["items"])
        self.assertIn("红细胞沉降率（ESR）", result["items"])
        self.assertIn("C反应蛋白（CRP）", result["items"])
        self.assertIn("凝血功能全套", result["items"])
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

    def test_conduction_path_uses_ecg_and_holter_not_low_magnesium_package_only(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["近晕厥", "头晕", "心动过缓"]},
            candidate_diseases=["二度房室传导阻滞"],
            proposed_items=["电解质"],
            existing_results={},
        )
        self.assertIn("心电图（ECG）", result["items"])
        self.assertIn("动态心电图（Holter）", result["items"])
        self.assertIn("体格检查", result["items"])
        self.assertNotIn("血清电解质", result["items"])
        self.assertNotIn("肾功能检查（RFTs）", result["items"])
        self.assertTrue(result["strict_diagnosis_driven"])

    def test_acute_tympanitis_uses_otoscopy_not_microtia_package(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["急性耳痛", "耳鸣", "听力下降"]},
            candidate_diseases=["急性鼓膜炎"],
            proposed_items=["听性脑干反应", "颞骨CT"],
            existing_results={},
        )
        self.assertIn("耳镜检查", result["items"])
        self.assertIn("体格检查", result["items"])
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertNotIn("听性脑干反应（ABR）", result["items"])
        self.assertNotIn("颞骨CT扫描（颞骨CT）", result["items"])
        self.assertTrue(result["strict_diagnosis_driven"])

    def test_crigler_najjar_path_uses_bilirubin_genetic_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["黄疸", "巩膜黄染", "嗜睡"]},
            candidate_diseases=["克里格勒-纳贾尔综合征"],
            proposed_items=["血常规"],
            existing_results={},
        )
        self.assertIn("肝功能检查（LFTs）", result["items"])
        self.assertIn("凝血功能全套", result["items"])
        self.assertIn("腹部超声", result["items"])
        self.assertIn("基因检测", result["items"])

    def test_chronic_nasopharyngitis_path_uses_ent_exams(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=4)
        result = strategy.recommend(
            collected_info={"symptoms": ["咽部异物感", "咽干", "反复清嗓"]},
            candidate_diseases=["慢性鼻咽炎"],
            proposed_items=["胸部X线"],
            existing_results={},
        )
        self.assertIn("鼻咽镜检查", result["items"])
        self.assertIn("脱落细胞学检查", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"][:2])

    def test_microtia_path_uses_abr_temporal_ct_and_gene_test(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=5)
        result = strategy.recommend(
            collected_info={"symptoms": ["出生即有小耳", "耳廓畸形", "听力下降"]},
            candidate_diseases=["小耳畸形"],
            proposed_items=["胸部X线"],
            existing_results={},
        )
        self.assertIn("听性脑干反应（ABR）", result["items"])
        self.assertIn("颞骨CT扫描（颞骨CT）", result["items"])
        self.assertIn("基因检测", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"][:3])

    def test_acute_bacterial_prostatitis_path_uses_dre_culture_ast(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={"symptoms": ["发热", "寒战", "尿频", "尿急", "尿痛", "会阴痛"]},
            candidate_diseases=["急性细菌性前列腺炎"],
            proposed_items=["尿动力学"],
            existing_results={},
        )
        self.assertIn("直肠指检（DRE）", result["items"])
        self.assertIn("尿液分析（UA）", result["items"])
        self.assertIn("尿培养", result["items"])
        self.assertIn("抗菌药物敏感性试验（AST）", result["items"])
        self.assertIn("前列腺超声", result["items"])
        self.assertNotIn("尿动力学检查（UDS）", result["items"][:6])

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
