import copy
import unittest

import yaml

from agent.agent import MyDoctorAgent
from hospital_agent.base import summarize_training_results


class FakeLLM:
    def __init__(self, result, metadata=None, *, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc
        self.last_call_metadata = metadata or {
            "model": "fake-model",
            "model_invoked": True,
            "attempt_index": 1,
            "http_status": 200,
            "latency_ms": 12.0,
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "finish_reason": "stop",
            "raw_response_present": True,
            "response_chars": 20,
            "exception_type": "",
        }
        self.model_name = "fake-model"

    async def chat_json(self, messages, temperature=None):
        if self.raise_exc:
            raise self.raise_exc
        return self.result

    async def chat(self, messages, temperature=None):
        if self.raise_exc:
            raise self.raise_exc
        return str(self.result)


def make_agent():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["memory"]["json_path"] = "tests/_runtime_chain/memory.json"
    config["memory"]["md_path"] = "tests/_runtime_chain/memory.md"
    config["memory"]["diagnostic_replay_path"] = "tests/_runtime_chain/replay.jsonl"
    config["policy_store_path"] = "tests/_runtime_chain/policies.json"
    config["self_improve_enabled"] = False
    return MyDoctorAgent(config)


class LLMCallAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_skip_creates_audit_record(self):
        agent = make_agent()
        agent.max_llm_calls_per_case = 1
        agent._llm_call_count = 1

        result = await agent._llm_chat_json([], purpose="thinking")

        self.assertEqual(result, {})
        record = agent._llm_call_audit[-1]
        self.assertFalse(record["model_invoked"])
        self.assertEqual(record["primary_failure_reason"], "llm_budget_exhausted")
        self.assertTrue(record["fallback_used"])

    async def test_truncated_json_parse_failure_keeps_generation_root_cause(self):
        agent = make_agent()
        agent.llm = FakeLLM(
            {"raw_response": '{"diagnosis": ['},
            metadata={
                "model": "fake-model",
                "model_invoked": True,
                "attempt_index": 1,
                "http_status": 200,
                "latency_ms": 15.0,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "finish_reason": "length",
                "raw_response_present": True,
                "response_chars": 15,
                "exception_type": "",
            },
        )

        await agent._llm_chat_json([], purpose="diagnosis")

        record = agent._llm_call_audit[-1]
        self.assertFalse(record["parse_success"])
        self.assertIn("generation_truncated", record["failure_flags"])
        self.assertIn("json_parse_failed", record["failure_flags"])
        self.assertEqual(record["primary_failure_reason"], "generation_truncated")

    async def test_schema_missing_fields_are_audited(self):
        agent = make_agent()
        agent.llm = FakeLLM({"not_diagnosis": []})

        await agent._llm_chat_json([], purpose="diagnosis")

        record = agent._llm_call_audit[-1]
        self.assertTrue(record["parse_success"])
        self.assertFalse(record["schema_success"])
        self.assertEqual(record["missing_fields"], ["diagnosis"])
        self.assertIn("schema_missing_fields", record["failure_flags"])

    def test_training_summary_aggregates_llm_contract_metrics(self):
        rows = [
            {
                "status": "evaluated",
                "audit": {
                    "llm_call_audit": [
                        {
                            "purpose": "diagnosis",
                            "json_expected": True,
                            "model_invoked": True,
                            "raw_response_present": True,
                            "parse_success": False,
                            "schema_applicable": False,
                            "failure_flags": ["json_parse_failed"],
                            "primary_failure_reason": "json_parse_failed",
                            "fallback_used": True,
                        },
                        {
                            "purpose": "thinking",
                            "json_expected": True,
                            "model_invoked": True,
                            "raw_response_present": True,
                            "parse_success": True,
                            "schema_applicable": True,
                            "schema_success": False,
                            "failure_flags": ["schema_missing_fields"],
                            "primary_failure_reason": "schema_missing_fields",
                            "fallback_used": True,
                        },
                    ]
                },
            }
        ]

        summary = summarize_training_results(rows)

        self.assertEqual(summary["llm_call_count"], 2)
        self.assertEqual(summary["llm_call_count_by_purpose"]["diagnosis"], 1)
        self.assertEqual(summary["llm_failure_count_by_reason"]["json_parse_failed"], 1)
        self.assertEqual(summary["llm_parse_failure_rate"], 0.5)
        self.assertEqual(summary["llm_schema_failure_rate"], 1.0)
        self.assertEqual(summary["llm_fallback_case_rate"], 1.0)
