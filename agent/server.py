"""
Agent HTTP 服务模块。

在端口 7860 上启动 HTTP 服务，暴露 POST /test 接口，
供 ModelScope Studio 平台调用进行测试。
"""

import asyncio
import json
import logging
import os
import traceback
from copy import deepcopy
from typing import Any, Dict, List

from aiohttp import web

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {
    "contestServiceToken",
    "serviceToken",
    "token",
    "api_key",
    "apiKey",
    "MODEL_API_KEY",
    "SERVICE_TRAIN_TOKEN",
    "authorization",
    "Authorization",
}


def _default_final_result(reason: str = "") -> Dict[str, Any]:
    """返回评测器可解析的兜底结果。"""
    return {
        "diagnosis": ["待明确诊断"],
        "treatment_plan": "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。",
        "reasoning": reason or "Agent 未能生成完整诊疗结果，返回兜底可评估结构。",
        "conversation_rounds": 0,
    }


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _find_first_value(value: Any, keys: tuple, depth: int = 0) -> Any:
    """递归查找请求体中的服务字段，兼容平台包在 input/data 的情况。"""
    if depth > 4:
        return None
    if isinstance(value, dict):
        for key in keys:
            if value.get(key):
                return value[key]
        for item in value.values():
            found = _find_first_value(item, keys, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_value(item, keys, depth + 1)
            if found:
                return found
    return None


def _normalize_final_result(value: Any, reason: str = "") -> Dict[str, Any]:
    """把不同来源的结果整理成评测器需要的稳定字段。"""
    if not isinstance(value, dict):
        return _default_final_result(reason)

    normalized = dict(value)
    diagnosis = (
        _as_str_list(value.get("diagnosis"))
        or _as_str_list(value.get("diagnoses"))
        or _as_str_list(value.get("final_diagnosis"))
        or _as_str_list(value.get("finalDiagnosis"))
    )
    if not diagnosis:
        diagnosis = ["待明确诊断"]

    treatment_plan = (
        value.get("treatment_plan")
        or value.get("treatment")
        or value.get("plan")
        or "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。"
    )

    normalized["diagnosis"] = diagnosis
    normalized["treatment_plan"] = str(treatment_plan)
    normalized["reasoning"] = str(
        value.get("reasoning")
        or value.get("reason")
        or reason
        or "已返回兜底诊疗结果。"
    )
    try:
        normalized["conversation_rounds"] = int(value.get("conversation_rounds", 0) or 0)
    except (TypeError, ValueError):
        normalized["conversation_rounds"] = 0
    return normalized


def _extract_patient_ids(body: Dict[str, Any]) -> List[str]:
    """兼容平台可能传入的 patient_id / caseId / cases 等字段。"""
    id_keys = ("patient_id", "patientId", "case_id", "caseId", "id")
    list_keys = ("patient_ids", "patientIds", "case_ids", "caseIds", "ids")

    patient_ids: List[str] = []
    for key in id_keys:
        if body.get(key):
            patient_ids.append(str(body[key]))

    for key in list_keys:
        for item in _as_str_list(body.get(key)):
            patient_ids.append(item)

    cases = body.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                for key in id_keys:
                    if case.get(key):
                        patient_ids.append(str(case[key]))
                        break
            elif case is not None:
                patient_ids.append(str(case))

    # 有些平台会把病例包在 input/data 里。
    for wrapper_key in ("input", "inputs", "data"):
        wrapped = body.get(wrapper_key)
        if isinstance(wrapped, dict):
            patient_ids.extend(_extract_patient_ids(wrapped))
        elif isinstance(wrapped, list):
            for item in wrapped:
                if isinstance(item, dict):
                    patient_ids.extend(_extract_patient_ids(item))
                elif item is not None:
                    patient_ids.append(str(item))

    return list(dict.fromkeys(patient_ids))


def _redact_for_log(value: Any, depth: int = 0) -> Any:
    """生成适合日志记录的脱敏摘要，避免泄露 token 和病例详情。"""
    if depth > 2:
        return "<nested>"
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key) in _SENSITIVE_KEYS:
                redacted[key] = "<redacted>"
            elif key in {"config", "input", "inputs", "data", "cases"}:
                redacted[key] = "<omitted>"
            else:
                redacted[key] = _redact_for_log(item, depth + 1)
        return redacted
    if isinstance(value, list):
        return [_redact_for_log(item, depth + 1) for item in value[:5]]
    if isinstance(value, str) and len(value) > 80:
        return value[:80] + "..."
    return value


def _apply_allowed_config_overrides(config: Dict[str, Any], override: Any) -> Dict[str, Any]:
    """只接受安全白名单配置覆盖，避免请求改写文件路径或外部服务地址。"""
    merged = deepcopy(config)
    if not isinstance(override, dict):
        return merged

    if isinstance(override.get("test"), dict):
        test_override = override["test"]
        allowed_test = {}
        if "patient_ids" in test_override:
            allowed_test["patient_ids"] = _as_str_list(test_override.get("patient_ids"))
        if "patient_count" in test_override:
            try:
                allowed_test["patient_count"] = max(1, int(test_override["patient_count"]))
            except (TypeError, ValueError):
                pass
        if allowed_test:
            merged.setdefault("test", {}).update(allowed_test)

    for key in ("max_ask_rounds", "max_exam_rounds"):
        if key in override:
            try:
                merged[key] = max(1, int(override[key]))
            except (TypeError, ValueError):
                logger.warning("[Server] 忽略非法配置覆盖: %s=%r", key, override[key])

    ignored_keys = sorted(set(override.keys()) - {"test", "max_ask_rounds", "max_exam_rounds"})
    if ignored_keys:
        logger.warning("[Server] 已忽略非白名单配置覆盖: %s", ignored_keys)

    return merged


def create_app() -> web.Application:
    """创建 aiohttp 应用。

    Returns:
        配置好路由的 aiohttp Application
    """
    app = web.Application()
    app.add_routes([
        web.post("/", handle_test),
        web.post("/test", handle_test),
        web.get("/health", handle_health),
        web.get("/", handle_health),
    ])
    return app


async def handle_health(request: web.Request) -> web.Response:
    """健康检查端点。

    Args:
        request: HTTP 请求

    Returns:
        健康状态响应
    """
    return web.json_response({"status": "ok", "service": "hospital-agent"})


async def handle_test(request: web.Request) -> web.Response:
    """处理 POST /test 请求。

    平台调用此接口触发 Agent 测试流程。
    请求体可包含：
    - patient_id: 单个患者 ID（可选）
    - patient_ids: 患者ID列表（可选）
    - config: 覆盖配置（可选）

    Args:
        request: HTTP 请求

    Returns:
        测试结果响应
    """
    try:
        # 解析请求体
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {"data": body}

        requested_patient_ids = _extract_patient_ids(body)
        logger.info(
            "[Server] 收到测试请求: path=%s, patient_ids=%s, body=%s",
            request.path,
            requested_patient_ids,
            json.dumps(_redact_for_log(body), ensure_ascii=False) if body else "empty",
        )

        # 加载配置
        import yaml
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # 用请求体中的白名单字段覆盖配置
        if body.get("config"):
            config = _apply_allowed_config_overrides(config, body["config"])

        # 如果请求指定了患者/病例 ID，覆盖配置。
        if requested_patient_ids:
            config.setdefault("test", {})["patient_ids"] = requested_patient_ids

        # 请求级服务凭证：只作用于当前 Agent 实例，不污染全局环境变量。
        runtime_service = config.setdefault("_runtime_service", {})
        request_token = _find_first_value(
            body,
            (
                "contestServiceToken",
                "contest_service_token",
                "serviceToken",
                "service_token",
            ),
        )
        request_team_id = _find_first_value(
            body,
            ("teamId", "team_id", "TEAM_ID"),
        )
        if request_token:
            runtime_service["token"] = str(request_token)
            logger.info("[Server] 使用请求级服务 token")

        if request_team_id:
            runtime_service["team_id"] = str(request_team_id)
            logger.info("[Server] 使用请求级 team id")

        # 创建 Agent 实例并运行测试
        from agent.agent import MyDoctorAgent
        agent = MyDoctorAgent(config)

        # 运行测试（确保资源清理）
        try:
            run_info = await agent.run_test()
        finally:
            await agent._cleanup()

        results = []
        test_dir = ""
        results_file = ""
        if isinstance(run_info, dict):
            results = run_info.get("results", []) or []
            test_dir = run_info.get("test_dir", "") or ""
            results_file = run_info.get("results_file", "") or ""

        # 关键：评测器 (batch_evaluation) 在响应顶层找 final_result / final_results。
        # 把 results 中每个 case 的 final_result 提到顶层，兼容单条/多条场景。
        _final_results_top: List[Dict[str, Any]] = []
        for _r in results:
            if isinstance(_r, dict) and isinstance(_r.get("final_results"), list) and _r["final_results"]:
                _final_results_top.extend(
                    _normalize_final_result(item, _r.get("error", ""))
                    for item in _r["final_results"]
                )
            elif isinstance(_r, dict) and _r.get("final_result") is not None:
                _final_results_top.append(_normalize_final_result(_r["final_result"], _r.get("error", "")))
            else:
                # 兜底空结构，保证评测器一定能取到可解析字段
                _final_results_top.append(
                    _default_final_result(_r.get("error", "") if isinstance(_r, dict) else "")
                )

        if not _final_results_top:
            _final_results_top = [_default_final_result("no results")]

        for idx, final_result in enumerate(_final_results_top):
            if requested_patient_ids:
                patient_id = requested_patient_ids[min(idx, len(requested_patient_ids) - 1)]
                final_result.setdefault("patient_id", patient_id)
                final_result.setdefault("caseId", patient_id)

        _final_result_single = _final_results_top[-1]

        response_data = {
            "status": "success",
            "message": "Test completed",
            "results": results,
            "result_count": len(results),
            "test_dir": test_dir,
            "results_file": results_file,
            # ↓↓↓ 评测器约定字段：顶层必须直接可读 ↓↓↓
            "final_result": _final_result_single,
            "final_results": _final_results_top,
        }
        if requested_patient_ids:
            response_data["patient_id"] = requested_patient_ids[0]
            response_data["caseId"] = requested_patient_ids[0]

        logger.info(f"[Server] 测试完成, 结果数={len(results)}, final_results顶层字段已注入")

        return web.json_response(response_data)

    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"[Server] {error_msg}\n{error_trace}")
        public_error = "Test failed; see server logs."

        return web.json_response(
            {
                "status": "error",
                "message": public_error,
                "final_result": _default_final_result(public_error),
                "final_results": [_default_final_result(public_error)],
            },
        )


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    """启动 HTTP 服务。

    Args:
        host: 监听地址
        port: 监听端口
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"[Server] 启动 Hospital Agent 服务, 监听 {host}:{port}")

    app = create_app()
    web.run_app(app, host=host, port=port, print=logger.info)
