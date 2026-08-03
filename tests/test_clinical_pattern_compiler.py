import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer, EvidenceBundle, Observation
from agent.clinical_pattern_compiler import ClinicalPatternCompiler
from agent.diagnosis_engine import DiagnosisDecisionEngine


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ClinicalPatternCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalizer = ClinicalEvidenceNormalizer("data/ref_data")
        cls.compiler = ClinicalPatternCompiler("data/ref_data")
        cls.engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")

    def test_mucocutaneous_platelet_bleeding_pattern_from_symptoms(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["皮肤瘀点、皮肤瘀斑，伴反复鼻出血和牙龈出血"]},
            {},
        )
        patterns = self.compiler.compile(evidence)
        pattern_ids = {item.pattern_id for item in patterns}
        self.assertIn("mucocutaneous_platelet_bleeding_pattern", pattern_ids)
        pattern = next(item for item in patterns if item.pattern_id == "mucocutaneous_platelet_bleeding_pattern")
        self.assertTrue(pattern.verified)
        self.assertIn("petechiae", pattern.supporting_findings)
        self.assertIn("epistaxis", pattern.supporting_findings)
        self.assertIn("thrombocytopenic_disorders", pattern.family_ids)

    def test_reactive_arthritis_pattern_from_triads(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["关节痛伴尿痛，眼红，查体提示结膜充血"]},
            {},
        )
        patterns = self.compiler.compile(evidence)
        pattern = next(
            item
            for item in patterns
            if item.pattern_id == "postinfectious_arthritis_uroocular_pattern"
        )
        self.assertGreaterEqual(pattern.confidence, 0.62)
        self.assertIn("postinfectious_immune_inflammation", pattern.mechanism_ids)
        self.assertIn("reactive_arthritis_spectrum", pattern.family_ids)

    def test_deep_bleeding_contradiction_prevents_platelet_pattern(self):
        evidence = EvidenceBundle(
            [
                Observation("epistaxis", "history", confidence=0.85),
                Observation("hemarthrosis", "history", confidence=0.88),
                Observation("pt_prolonged", "coagulation", confidence=0.92),
            ]
        )
        patterns = self.compiler.compile(evidence)
        self.assertNotIn(
            "mucocutaneous_platelet_bleeding_pattern",
            {item.pattern_id for item in patterns},
        )

    def test_ordinary_uti_with_nonspecific_leg_pain_does_not_form_reiter_pattern(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["尿痛尿频，伴腿痛，但没有眼红"]},
            {},
        )
        patterns = self.compiler.compile(evidence)
        self.assertNotIn(
            "postinfectious_arthritis_uroocular_pattern",
            {item.pattern_id for item in patterns},
        )

    def test_itp_enters_top5_from_platelet_bleeding_pattern(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["皮肤瘀点、皮肤瘀斑，伴反复鼻出血和牙龈出血"]},
            {},
        )
        decision = self.engine.decide({}, [], evidence)
        top5 = [item.diagnosis for item in decision.candidates[:5]]
        self.assertIn("特发性血小板减少性紫癜", top5)
        itp = next(item for item in decision.candidates if item.diagnosis == "特发性血小板减少性紫癜")
        self.assertEqual(itp.eligibility_status, "Deferred")
        self.assertTrue(
            any(
                source.get("source") == "clinical_pattern"
                and source.get("metadata", {}).get("pattern_id") == "mucocutaneous_platelet_bleeding_pattern"
                for source in itp.candidate_sources
            )
        )

    def test_reiter_enters_top5_from_arthritis_uroocular_pattern(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["关节痛伴尿痛、尿道刺激症状，眼红并有结膜充血"]},
            {},
        )
        decision = self.engine.decide({}, [], evidence)
        top5 = [item.diagnosis for item in decision.candidates[:5]]
        self.assertIn("赖特综合征", top5)
        reiter = next(item for item in decision.candidates if item.diagnosis == "赖特综合征")
        self.assertTrue(
            any(
                source.get("source") == "clinical_pattern"
                and source.get("metadata", {}).get("pattern_id") == "postinfectious_arthritis_uroocular_pattern"
                for source in reiter.candidate_sources
            )
        )


if __name__ == "__main__":
    unittest.main()
