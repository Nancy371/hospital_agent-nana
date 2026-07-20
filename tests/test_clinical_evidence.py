import unittest

from agent.clinical_evidence import ClinicalEvidenceNormalizer, EvidenceAgent


class ClinicalEvidenceNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = ClinicalEvidenceNormalizer("data/ref_data")

    def test_local_negation_does_not_leak_to_next_clause(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {
                        "结论": "未见先天性缺损（如 ASD、VSD），但见重度二尖瓣反流。"
                    },
                }
            },
        )
        asd = [item for item in bundle.observations if item.finding == "atrial_septal_defect"]
        asd_dx = [item for item in bundle.observations if item.finding == "diagnosis:房间隔缺损"]
        mitral = [item for item in bundle.observations if item.finding == "mitral_regurgitation"]
        self.assertTrue(asd and all(item.polarity == "negative" for item in asd))
        self.assertTrue(asd_dx and all(item.polarity == "negative" for item in asd_dx))
        self.assertTrue(mitral and any(item.polarity == "positive" for item in mitral))

    def test_numeric_value_and_reference_range_create_low_magnesium(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            },
        )
        hits = [item for item in bundle.observations if item.finding == "low_magnesium"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].direction, "low")
        self.assertAlmostEqual(hits[0].value, 0.45)

    def test_urine_magnesium_uses_leaf_value_not_field_number(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "24小时尿电解质检测": {
                    "status": "abnormal",
                    "result": {
                        "24小时尿镁": "1.2 mmol/24h［参考范围：3.0-5.0］",
                    },
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:24小时尿镁")
        self.assertAlmostEqual(field.value, 1.2)
        self.assertEqual(field.direction, "low")
        self.assertEqual(field.unit, "mmol/24h")
        self.assertIn("low_urine_magnesium", bundle.findings("positive"))
        self.assertIn("magnesium_depletion", bundle.findings("positive"))
        self.assertNotIn("low_magnesium", bundle.findings("positive"))

    def test_magnesium_load_retention_with_textual_upper_range_is_high(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {
                        "镁负荷保留率": "62%［参考范围：镁储备充足时通常＜20-30%］",
                    },
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:镁负荷保留率")
        self.assertAlmostEqual(field.value, 62.0)
        self.assertEqual(field.direction, "high")
        self.assertIn("magnesium_load_retention_high", bundle.findings("positive"))
        self.assertIn("magnesium_depletion", bundle.findings("positive"))

    def test_reference_only_magnesium_range_has_no_numeric_disease_evidence(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "报告模板": {
                    "status": "normal",
                    "result": {"备注": "血镁参考范围：0.75-1.02 mmol/L"},
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:备注")
        self.assertIsNone(field.value)
        self.assertNotIn("low_magnesium", bundle.findings("positive"))
        self.assertNotIn("magnesium_depletion", bundle.findings("positive"))

    def test_reference_example_does_not_become_positive_diagnosis(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "报告模板": {
                    "status": "normal",
                    "result": {"备注": "参考范围：ASD 仅用于报告模板示例"},
                }
            },
        )
        positives = {
            item.finding for item in bundle.observations if item.polarity == "positive"
        }
        self.assertNotIn("atrial_septal_defect", positives)
        self.assertNotIn("diagnosis:房间隔缺损", positives)

    def test_vasculitis_serology_and_red_cell_casts_are_structured(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "抗中性粒细胞胞质抗体（ANCA）谱": {
                    "status": "abnormal",
                    "result": {"髓过氧化物酶抗体": "MPO抗体阳性"},
                },
                "尿液分析（UA）": {
                    "status": "abnormal",
                    "result": {"尿沉渣": "可见红细胞管型"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("mpo_anca_positive", findings)
        self.assertIn("microscopic_hematuria", findings)

    def test_bronchopneumonia_imaging_terms_are_structured(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "胸部CT扫描（Chest CT）": {
                    "status": "abnormal",
                    "result": {"结论": "右下叶实变并见空气支气管征"},
                },
                "支气管镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "支气管内见脓性分泌物"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("pneumonia_infiltrate", findings)
        self.assertIn("bronchopneumonia", findings)

    def test_short_mr_alias_does_not_match_mri(self):
        bundle = self.normalizer.normalize(
            {},
            {"影像": {"status": "normal", "result": {"结论": "MRI 未见异常"}}},
        )
        self.assertFalse(
            any(item.finding == "diagnosis:二尖瓣反流" for item in bundle.observations)
        )

    def test_evidence_agent_builds_graph_categories(self):
        graph = EvidenceAgent(normalizer=self.normalizer).build_graph(
            {"symptoms": ["腹胀", "下腹部不适"], "history": "近期促排卵后取卵"},
            {
                "盆腔超声": {
                    "status": "abnormal",
                    "result": {"卵巢": "双侧卵巢增大，伴腹水"},
                },
                "血常规": {
                    "status": "abnormal",
                    "result": {"红细胞压积": "48%［参考值：35-45%］"},
                },
            },
        )
        findings = {item["finding"] for item in graph.observations}
        self.assertIn("ohss_risk", findings)
        self.assertIn("ovarian_enlargement", findings)
        self.assertIn("ascites", findings)
        self.assertIn("hemoconcentration", findings)
        self.assertTrue(graph.symptoms)
        self.assertTrue(graph.imaging)
        self.assertTrue(graph.labs)
        self.assertTrue(graph.risk_factors)


if __name__ == "__main__":
    unittest.main()
