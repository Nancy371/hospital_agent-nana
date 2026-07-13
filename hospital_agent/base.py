"""
Hospital Agent SDK 基类实现。

提供 BaseDoctorAgent 基类和 Actions 接口，
通过 HTTP 调用比赛服务 API 实现问诊、检查、诊疗等能力。
"""

import json
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Actions:
    """比赛能力接口，封装与服务端的 HTTP 交互。

    提供 5 个核心 Action：
    - ask_patient：询问患者
    - order_examination：申请检查
    - prescribe_treatment：提交诊断和治疗方案
    - evaluation：训练阶段获取评测结果
    - batch_evaluation：批量评估
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        team_id: str,
        endpoint_prefixes: Optional[List[str]] = None,
    ):
        """初始化 Actions。

        Args:
            base_url: 服务基础 URL
            token: 认证 token
            team_id: 团队 ID
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.team_id = team_id
        self.endpoint_prefixes = self._normalize_endpoint_prefixes(endpoint_prefixes or [])
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _normalize_endpoint_prefixes(prefixes: List[str]) -> List[str]:
        normalized = [""]
        for prefix in prefixes or []:
            text = str(prefix or "").strip()
            if not text:
                continue
            if not text.startswith("/"):
                text = "/" + text
            text = text.rstrip("/")
            if text not in normalized:
                normalized.append(text)
        return normalized

    def _candidate_paths(self, path: str) -> List[str]:
        clean_path = "/" + str(path or "").lstrip("/")
        paths: List[str] = []
        for prefix in self.endpoint_prefixes:
            candidate = f"{prefix}{clean_path}" if prefix else clean_path
            if candidate not in paths:
                paths.append(candidate)
        return paths

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端。"""
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("SERVICE_BASE_URL must include http:// or https://")
        if not self.token:
            raise ValueError("SERVICE_TRAIN_TOKEN is required")
        if not self.team_id:
            raise ValueError("TEAM_ID is required")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "X-Team-ID": self.team_id,
                },
                timeout=60.0,
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """发送 HTTP 请求。

        Args:
            method: HTTP 方法
            path: API 路径
            **kwargs: 传递给 httpx 的额外参数

        Returns:
            响应 JSON 数据

        Raises:
            httpx.HTTPError: HTTP 请求失败
        """
        client = await self._get_client()
        last_error: Optional[httpx.HTTPStatusError] = None
        candidate_paths = self._candidate_paths(path)
        tried_paths: List[str] = []
        for candidate_path in candidate_paths:
            tried_paths.append(candidate_path)
            response = await client.request(method, candidate_path, **kwargs)
            try:
                response.raise_for_status()
                if candidate_path != path:
                    logger.info(
                        "[Action] endpoint fallback 命中: %s -> %s",
                        path,
                        candidate_path,
                    )
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 404 and candidate_path != candidate_paths[-1]:
                    logger.warning(
                        "[Action] %s %s 返回 404，尝试下一个 endpoint 前缀",
                        method,
                        candidate_path,
                    )
                    continue
                break

        if last_error is not None:
            response = last_error.response
            detail = response.text[:300] if response is not None else ""
            if response is not None and response.status_code == 404:
                logger.error(
                    "[Action] 404 Not Found: base_url=%s, tried_paths=%s, "
                    "可能原因：病例 ID 在当前 token/team 下不可访问，或 SERVICE_BASE_URL/endpoint 前缀不匹配。"
                    "response=%s",
                    self.base_url,
                    tried_paths,
                    detail,
                )
            raise last_error
        raise RuntimeError(f"HTTP request failed before sending: {method} {path}")

    async def ask_patient(
        self, patient_id: str, input_data: Dict[str, Any]
    ) -> str:
        """询问患者。

        Args:
            patient_id: 患者 ID
            input_data: 输入数据，包含 question 和 chat_history

        Returns:
            患者的回复文本
        """
        logger.info(f"[Action] ask_patient: patient_id={patient_id}")
        try:
            result = await self._request(
                "POST",
                "/ask_patient",
                json={
                    "patient_id": patient_id,
                    "input_data": input_data,
                    "team_id": self.team_id,
                },
            )
            answer = result.get("answer", result.get("response", ""))
            if isinstance(answer, dict):
                answer = json.dumps(answer, ensure_ascii=False)
            return str(answer)
        except Exception as e:
            logger.error(f"[Action] ask_patient 失败: {e}")
            raise

    async def order_examination(
        self,
        patient_id: str,
        items: List[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        """申请检查。

        Args:
            patient_id: 患者 ID
            items: 检查项目列表
            reason: 申请原因

        Returns:
            检查结果字典，包含 "results" 键
        """
        logger.info(
            f"[Action] order_examination: patient_id={patient_id}, items={items}"
        )
        try:
            result = await self._request(
                "POST",
                "/order_examination",
                json={
                    "patient_id": patient_id,
                    "items": items,
                    "reason": reason,
                    "team_id": self.team_id,
                },
            )
            return result
        except Exception as e:
            logger.error(f"[Action] order_examination 失败: {e}")
            raise

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: List[str],
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """提交诊断和治疗方案。

        Args:
            patient_id: 患者 ID
            diagnosis: 诊断列表
            treatment_plan: 治疗方案
            reasoning: 诊断推理

        Returns:
            提交结果
        """
        logger.info(
            f"[Action] prescribe_treatment: patient_id={patient_id}, diagnosis={diagnosis}"
        )
        try:
            result = await self._request(
                "POST",
                "/prescribe_treatment",
                json={
                    "patient_id": patient_id,
                    "diagnosis": diagnosis,
                    "treatment_plan": treatment_plan,
                    "reasoning": reasoning,
                    "team_id": self.team_id,
                },
            )
            return result
        except Exception as e:
            logger.error(f"[Action] prescribe_treatment 失败: {e}")
            raise

    async def evaluation(
        self,
        patient_id: str,
        final_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """训练阶段获取评测结果。

        Args:
            patient_id: 患者 ID
            final_result: 最终诊疗结果

        Returns:
            评测报告字典
        """
        logger.info(f"[Action] evaluation: patient_id={patient_id}")
        try:
            result = await self._request(
                "POST",
                "/evaluation",
                json={
                    "patient_id": patient_id,
                    "final_result": final_result,
                    "team_id": self.team_id,
                },
            )
            return result
        except Exception as e:
            logger.error(f"[Action] evaluation 失败: {e}")
            raise

    async def batch_evaluation(self, test_dir: str) -> Dict[str, Any]:
        """批量评估。

        Args:
            test_dir: 测试结果目录路径

        Returns:
            批量评估报告
        """
        logger.info(f"[Action] batch_evaluation: test_dir={test_dir}")
        try:
            result = await self._request(
                "POST",
                "/batch_evaluation",
                json={
                    "test_dir": test_dir,
                    "team_id": self.team_id,
                },
            )
            return result
        except Exception as e:
            logger.error(f"[Action] batch_evaluation 失败: {e}")
            raise

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class BaseDoctorAgent(ABC):
    """医生 Agent 基类。

    参赛者需要继承此类并实现 train 和 test 方法。
    基类提供：
    - self.actions: Actions 实例，用于调用比赛能力
    - self.config: 配置字典
    - run_train(): 训练入口
    - run_test(): 测试入口
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 Agent。

        Args:
            config: 配置字典
        """
        self.config = config

        # 从请求级配置 / 环境变量 / 配置文件获取服务配置，避免修改进程全局环境。
        runtime_service = config.get("_runtime_service", {}) or {}
        service_config = config.get("service", {}) or {}
        base_url = (
            runtime_service.get("base_url")
            or os.environ.get("SERVICE_BASE_URL", "")
            or service_config.get("base_url", "")
        )
        token = (
            runtime_service.get("token")
            or os.environ.get("SERVICE_TRAIN_TOKEN", "")
            or service_config.get("token", "")
        )
        team_id = (
            runtime_service.get("team_id")
            or os.environ.get("TEAM_ID", "")
            or service_config.get("team_id", "")
        )
        endpoint_prefixes = (
            runtime_service.get("endpoint_prefixes")
            or service_config.get("endpoint_prefixes")
            or []
        )
        if isinstance(endpoint_prefixes, str):
            endpoint_prefixes = [endpoint_prefixes]

        # 创建 Actions 实例
        self.actions = Actions(
            base_url=base_url,
            token=token,
            team_id=team_id,
            endpoint_prefixes=endpoint_prefixes,
        )

        # 输出目录
        self.output_dir = config.get("output_dir", "outputs")

    @abstractmethod
    async def train(self, patient_id: str) -> None:
        """训练流程：对单个患者进行诊疗。

        参赛者必须实现此方法。

        Args:
            patient_id: 患者 ID
        """
        ...

    @abstractmethod
    async def test(self, patient_id: str) -> None:
        """测试流程：对单个患者进行诊疗并提交结果。

        参赛者必须实现此方法。

        Args:
            patient_id: 患者 ID
        """
        ...

    async def _get_patient_list(
        self, mode: str = "train"
    ) -> List[str]:
        """从服务端获取患者列表。

        Args:
            mode: "train" 或 "test"

        Returns:
            患者 ID 列表
        """
        try:
            result = await self.actions._request(
                "GET",
                "/patients",
                params={"mode": mode, "team_id": self.actions.team_id},
            )
            patients = result.get("patients", result.get("data", []))
            return [p.get("patient_id", p) if isinstance(p, dict) else str(p) for p in patients]
        except Exception as e:
            logger.warning(f"获取患者列表失败: {e}，使用配置文件中的列表")
            return []

    def _get_patient_ids_from_config(self, mode: str = "train") -> List[str]:
        """从配置文件获取患者 ID 列表。

        Args:
            mode: "train" 或 "test"

        Returns:
            患者 ID 列表
        """
        mode_config = self.config.get(mode, {})
        patient_ids = mode_config.get("patient_ids", [])

        if patient_ids:
            return patient_ids

        # 如果没有指定患者 ID，返回空列表
        # 实际运行时会从服务端获取
        return []

    def _select_patient_ids(self, patient_ids: List[str], mode: str = "train") -> List[str]:
        """按 config.yaml 的 selection/patient_count/random_seed 选择患者。"""
        if not patient_ids:
            return []

        mode_config = self.config.get(mode, {}) or {}
        explicit_ids = mode_config.get("patient_ids", []) or []
        if explicit_ids:
            return [str(pid) for pid in explicit_ids]

        selection = str(mode_config.get("selection", "forward") or "forward").lower()
        patient_count = mode_config.get("patient_count")
        try:
            limit = int(patient_count) if patient_count not in (None, "") else len(patient_ids)
        except (TypeError, ValueError):
            limit = len(patient_ids)
        limit = max(0, min(limit, len(patient_ids)))

        selected = [str(pid) for pid in patient_ids]
        if selection == "random":
            seed = mode_config.get("random_seed", 42)
            rng = random.Random(seed)
            selected = selected[:]
            rng.shuffle(selected)
        elif selection == "reverse":
            selected = list(reversed(selected))
        elif selection != "forward":
            logger.warning("[%s] 未知 selection=%r，按 forward 处理", mode, selection)

        return selected[:limit] if limit else []

    async def _cleanup(self) -> None:
        """清理资源，关闭所有 HTTP 客户端。幂等方法，可安全多次调用。"""
        try:
            await self.actions.close()
        except Exception as e:
            logger.warning(f"[Cleanup] 关闭 actions 客户端失败: {e}")
        # 如果子类有 llm 客户端，也关闭它
        if hasattr(self, 'llm') and hasattr(self.llm, 'close'):
            try:
                await self.llm.close()
            except Exception as e:
                logger.warning(f"[Cleanup] 关闭 llm 客户端失败: {e}")

    async def run_train(self) -> None:
        """训练入口：获取患者列表并逐个训练。

        此方法由 train.py 调用。
        """
        logger.info("[run_train] 开始训练流程")

        # 获取患者列表
        patient_ids = self._get_patient_ids_from_config("train")

        if not patient_ids:
            # 尝试从服务端获取
            patient_ids = await self._get_patient_list("train")
        patient_ids = self._select_patient_ids(patient_ids, "train")

        if not patient_ids:
            # 使用配置中的数量
            train_config = self.config.get("train", {})
            patient_count = train_config.get("patient_count", 10)
            logger.warning(
                f"[run_train] 无法获取患者列表，"
                f"请确保 SERVICE_BASE_URL 正确或配置 patient_ids。"
                f"配置 patient_count={patient_count}"
            )
            return

        logger.info(f"[run_train] 训练患者数: {len(patient_ids)}")

        # 创建输出目录
        train_dir = os.path.join(self.output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # 逐个训练
        success_count = 0
        fail_count = 0
        for i, patient_id in enumerate(patient_ids):
            logger.info(f"[run_train] 训练进度: {i + 1}/{len(patient_ids)}, patient_id={patient_id}")
            try:
                await self.train(patient_id)
                success_count += 1
            except Exception as e:
                logger.error(f"[run_train] 训练患者 {patient_id} 失败: {e}")
                fail_count += 1

        logger.info(
            f"[run_train] 训练完成: 成功={success_count}, 失败={fail_count}"
        )

        # 清理资源
        await self._cleanup()

    async def run_test(self) -> Dict[str, Any]:
        """测试入口：获取患者列表并逐个测试。

        此方法由 test.py 调用。
        """
        logger.info("[run_test] 开始测试流程")

        # 获取患者列表
        patient_ids = self._get_patient_ids_from_config("test")

        if not patient_ids:
            # 尝试从服务端获取
            patient_ids = await self._get_patient_list("test")
        patient_ids = self._select_patient_ids(patient_ids, "test")

        if not patient_ids:
            logger.warning("[run_test] 无法获取患者列表，请确保配置正确")
            return {
                "test_dir": "",
                "results_file": "",
                "results": [],
                "success_count": 0,
                "fail_count": 0,
            }

        logger.info(f"[run_test] 测试患者数: {len(patient_ids)}")

        # 创建输出目录
        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        test_dir = os.path.join(self.output_dir, "test", timestamp)
        os.makedirs(test_dir, exist_ok=True)

        # 逐个测试
        results = []
        success_count = 0
        fail_count = 0
        for i, patient_id in enumerate(patient_ids):
            logger.info(f"[run_test] 测试进度: {i + 1}/{len(patient_ids)}, patient_id={patient_id}")
            try:
                await self.test(patient_id)
                success_count += 1
                # 收集测试结果（子类可在 test 中设置 self._last_test_result）
                result_entry = {"patient_id": patient_id, "status": "success"}
                if hasattr(self, '_last_test_result') and self._last_test_result:
                    result_entry.update(self._last_test_result)
                results.append(result_entry)
            except Exception as e:
                logger.error(f"[run_test] 测试患者 {patient_id} 失败: {e}")
                fail_count += 1
                # 失败也要给评测器一个可解析的 final_result 兜底，避免 "响应中未找到 final_result" 类错误
                _empty_final = {
                    "patient_id": patient_id,
                    "diagnosis": ["待明确诊断"],
                    "treatment_plan": "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。",
                    "reasoning": f"agent failed: {e}",
                    "conversation_rounds": 0,
                    "ordered_examinations": [],
                    "finished": True,
                }
                results.append({
                    "patient_id": patient_id,
                    "status": "failed",
                    "error": str(e),
                    "diagnosis": _empty_final["diagnosis"],
                    "treatment_plan": _empty_final["treatment_plan"],
                    "reasoning": _empty_final["reasoning"],
                    "conversation_rounds": 0,
                    "ordered_examinations": [],
                    "finished": True,
                    "final_result": _empty_final,
                    "final_results": [_empty_final],
                })

        # 保存测试结果
        results_file = os.path.join(test_dir, "final_results.jsonl")
        tmp_results_file = os.path.join(
            test_dir,
            f".final_results.{os.getpid()}.{uuid.uuid4().hex}.tmp",
        )
        with open(tmp_results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_results_file, results_file)
        except OSError:
            # 某些受限 Windows 沙盒禁止重命名替换，退化为直接写入以保证功能可用。
            with open(results_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(
            f"[run_test] 测试完成: 成功={success_count}, 失败={fail_count}"
        )
        logger.info(f"[run_test] 结果保存到: {results_file}")

        # 清理资源
        await self._cleanup()
        return {
            "test_dir": test_dir,
            "results_file": results_file,
            "results": results,
            "success_count": success_count,
            "fail_count": fail_count,
        }
