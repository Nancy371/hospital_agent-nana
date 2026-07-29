import unittest

from agent.exam_resolver import (
    ALIAS,
    PARTIAL_SUBSTITUTE,
    UNRESOLVED,
    ExamResolver,
)


class ExamResolverTests(unittest.TestCase):
    def test_alias_resolution_is_full_coverage(self):
        resolver = ExamResolver(
            catalog_names=["骨髓流式细胞免疫表型分析"],
            aliases={"流式细胞术免疫分型": "骨髓流式细胞免疫表型分析"},
        )

        result = resolver.resolve("流式细胞术免疫分型")

        self.assertEqual(result.resolution_type, ALIAS)
        self.assertEqual(result.resolved_exam, "骨髓流式细胞免疫表型分析")
        self.assertEqual(result.diagnostic_coverage, 1.0)

    def test_partial_substitute_is_not_full_confirmation(self):
        resolver = ExamResolver(catalog_names=["骨髓穿刺和活检（BMAB）", "全血细胞计数（CBC）"])

        result = resolver.resolve("流式细胞术免疫分型")

        self.assertEqual(result.resolution_type, PARTIAL_SUBSTITUTE)
        self.assertEqual(result.resolved_exam, "骨髓穿刺和活检（BMAB）")
        self.assertLess(result.diagnostic_coverage, 1.0)
        self.assertNotEqual(result.resolved_exam, "全血细胞计数（CBC）")

    def test_special_pulmonary_vascular_request_resolves_as_partial_if_only_ct_exists(self):
        resolver = ExamResolver(catalog_names=["胸部CT扫描（Chest CT）"])

        result = resolver.resolve("肺动脉CTA")

        self.assertEqual(result.resolution_type, PARTIAL_SUBSTITUTE)
        self.assertEqual(result.resolved_exam, "胸部CT扫描（Chest CT）")
        self.assertLess(result.diagnostic_coverage, 1.0)

    def test_unresolved_exam_stays_unresolved(self):
        resolver = ExamResolver(catalog_names=["血常规"])

        result = resolver.resolve("完全不存在的专科检查")

        self.assertEqual(result.resolution_type, UNRESOLVED)
        self.assertEqual(result.resolved_exam, "")


if __name__ == "__main__":
    unittest.main()
