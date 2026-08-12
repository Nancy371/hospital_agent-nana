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


class SequenceLLM(FakeLLM):
    def __init__(self, results):
        super().__init__({})
        self.results = list(results)
        self.calls = 0

    async def chat_json(self, messages, temperature=None):
        if not self.results:
            raise AssertionError("no more fake LLM responses")
        self.calls += 1
        self.last_call_metadata = {
            "model": "fake-model",
            "model_invoked": True,
            "attempt_index": self.calls,
            "http_status": 200,
            "latency_ms": 10.0,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "finish_reason": "stop",
            "raw_response_present": True,
            "response_chars": 20,
            "exception_type": "",
        }
        return self.results.pop(0)


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
    async def test_planner_consumer_result_uses_agent_audit_callback(self):
        agent = make_agent()
        agent.llm = FakeLLM({"unexpected": "shape"})
        planner = agent._get_planner()
        planner.criticism_max_calls = 0

        result = await planner.plan({}, {}, [])

        self.assertIn("strategy", result)
        record = agent._llm_call_audit[-1]
        self.assertEqual(record["purpose"], "planning")
        self.assertFalse(record["consumer_accepted"])
        self.assertTrue(record["fallback_used"])
        self.assertEqual(record["fallback_trigger"], "schema_missing_fields")

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

    async def test_contract_executor_repairs_missing_required_field_once(self):
        agent = make_agent()
        agent.llm = SequenceLLM(
            [
                {"not_diagnosis": ["x"]},
                {"diagnosis": ["A"], "treatment_plan": "observe", "reasoning": "fixed"},
            ]
        )

        result = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(result["diagnosis"], ["A"])
        records = agent._llm_call_audit
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["logical_call_id"], records[1]["logical_call_id"])
        self.assertEqual(records[0]["attempt_type"], "generate")
        self.assertEqual(records[1]["attempt_type"], "repair")
        self.assertTrue(records[1]["contract_repair_succeeded"])
        self.assertEqual(agent._llm_call_by_kind["json"], 1)
        self.assertEqual(agent._llm_call_by_kind["json_repair"], 1)

    async def test_contract_repair_respects_llm_budget(self):
        agent = make_agent()
        agent.max_llm_calls_per_case = 1
        agent.llm = SequenceLLM([{"not_diagnosis": ["x"]}])

        result = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(result, {"not_diagnosis": ["x"]})
        records = agent._llm_call_audit
        self.assertEqual(len(records), 2)
        self.assertTrue(records[1]["contract_repair_attempted"])
        self.assertFalse(records[1]["model_invoked"])
        self.assertEqual(records[1]["primary_failure_reason"], "llm_budget_exhausted")

    def test_stage_context_compiler_trims_and_audits_context(self):
        agent = make_agent()
        chat_history = [{"from": "doctor", "text": str(i)} for i in range(20)]

        compiled = agent._compile_llm_context(
            "thinking",
            collected_info={"symptoms": ["cough"]},
            exam_results={},
            chat_history=chat_history,
        )

        self.assertEqual(len(compiled["chat_history"]), 12)
        self.assertEqual(compiled["chat_history"][0]["text"], "8")
        audit = agent._llm_context_audit[-1]
        self.assertEqual(audit["stage"], "thinking")
        self.assertIn("chat_history:8_old_items", audit["omitted_sections"])
        self.assertGreater(audit["estimated_input_tokens"], 0)

    def test_training_summary_aggregates_llm_contract_metrics(self):
        rows = [
            {
                "status": "evaluated",
                "audit": {
                    "llm_context_audit": [
                        {
                            "stage": "diagnosis",
                            "context_chars": 400,
                            "estimated_input_tokens": 100,
                        }
                    ],
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
        self.assertEqual(summary["llm_context_compile_count"], 1)
        self.assertEqual(summary["llm_context_compile_count_by_stage"], {"diagnosis": 1})
        self.assertEqual(summary["average_llm_context_estimated_input_tokens"], 100.0)
