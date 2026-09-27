"""Bounded vLLM and optional local NVML telemetry.

``Telemetry`` owns one persistent HTTPX client for its lifetime. Call
``prepare()`` before the measured run so endpoint and NVML initialization
failures are captured before sampling begins. Missing Prometheus series are
reported as ``None`` (and listed under ``missing_metrics``), never as zero.
NVML values describe the machine running this Python process; they do not
identify GPU activity attributable to an individual request.
"""

from __future__ import annotations

import asyncio
import copy
import math
import time
from collections.abc import Mapping
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families


_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "running_requests": ("vllm:num_requests_running",),
    "waiting_requests": ("vllm:num_requests_waiting",),
    "prompt_tokens_total": (
        "vllm:prompt_tokens_total", "vllm:prompt_tokens",
    ),
    "generation_tokens_total": (
        "vllm:generation_tokens_total", "vllm:generation_tokens",
    ),
    "kv_cache_fraction": (
        "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
    ),
}
_KNOWN_METRIC_SAMPLE_NAMES = {
    name for metric_names in _METRIC_NAMES.values() for name in metric_names
}
_COUNTER_FIELDS = ("prompt_tokens_total", "generation_tokens_total")


def parse_vllm_metrics(text: str, model_name: str | None = None) -> dict[str, Any]:
    """Parse and aggregate known vLLM samples, optionally by ``model_name``.

    Request gauges and counters are summed across matching series. KV-cache
    usage is the maximum matching fraction because each series can describe a
    separate engine/replica. The historical ``gpu_cache_usage_perc`` name is
    accepted when the current ``kv_cache_usage_perc`` series is absent.
    """
    by_name: dict[str, list[float]] = {}
    invalid: list[dict[str, Any]] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in _KNOWN_METRIC_SAMPLE_NAMES:
                continue
            labels = sample.labels
            if model_name is not None and labels.get("model_name") != model_name:
                continue
            try:
                value = float(sample.value)
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            if not math.isfinite(value):
                invalid.append({
                    "name": sample.name,
                    "labels": dict(labels),
                    "value": str(sample.value),
                })
                continue
            by_name.setdefault(sample.name, []).append(value)

    values: dict[str, float | None] = {}
    missing: list[str] = []
    for field, names in _METRIC_NAMES.items():
        matched = next((by_name[name] for name in names if name in by_name), None)
        if not matched:
            values[field] = None
            missing.append(names[0])
        else:
            try:
                aggregate = (max(matched) if field == "kv_cache_fraction"
                             else math.fsum(matched))
            except (OverflowError, ValueError):
                aggregate = math.inf
            if math.isfinite(aggregate):
                values[field] = aggregate
            else:
                values[field] = None
                invalid.append({
                    "name": ",".join(names),
                    "labels": {},
                    "value": "non-finite aggregate",
                })
                missing.append(names[0])

    return {**values, "missing_metrics": missing, "invalid_metrics": invalid}


def counter_rates(
    current: Mapping[str, float | None],
    previous: Mapping[str, float | None] | None,
    elapsed: float,
) -> tuple[dict[str, float | None], list[str]]:
    """Return per-counter rates; the first sample and counter resets are null."""
    rates: dict[str, float | None] = {}
    resets: list[str] = []
    for field in _COUNTER_FIELDS:
        now = current.get(field)
        before = previous.get(field) if previous is not None else None
        if now is None or before is None or elapsed <= 0:
            rates[field] = None
            continue
        delta = now - before
        if delta < 0:
            rates[field] = None
            resets.append(field)
        else:
            rates[field] = delta / elapsed
    return rates, resets


class Telemetry:
    """Periodically sample a vLLM ``/metrics`` URL and optional local NVML.

    ``metrics_url`` must point to the Prometheus endpoint itself. ``interval``
    and ``timeout`` are bounded positive seconds. Set ``nvml_device`` to a
    local CUDA/NVML device index to request host-level GPU telemetry.
    """

    def __init__(
        self,
        metrics_url: str | None,
        model_name: str | None = None,
        nvml_device: int | None = None,
        interval: float = 0.2,
        timeout: float = 1.0,
    ) -> None:
        if interval <= 0 or not math.isfinite(interval):
            raise ValueError("interval must be a finite positive number")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a finite positive number")
        if nvml_device is not None and nvml_device < 0:
            raise ValueError("nvml_device must be non-negative")
        self.metrics_url = metrics_url
        self.model_name = model_name
        self.nvml_device = nvml_device
        self.interval = interval
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._prepared = False
        self._closed = False
        self._metrics_error: str | None = None
        self._nvml_error: str | None = None
        self._nvml: Any = None
        self._nvml_handle: Any = None
        self._nvml_initialized = False
        self._last_metrics: dict[str, Any] | None = None
        self._last_counters: dict[str, float | None] | None = None
        self._last_counter_time: float | None = None
        self._probe_receipt: dict[str, Any] | None = None
        self._probe_emitted = False

    async def _read_metrics(self) -> tuple[dict[str, Any] | None, float]:
        if self.metrics_url is None:
            return None, 0.0
        started = time.perf_counter()
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            response = await asyncio.wait_for(
                self._client.get(self.metrics_url), timeout=self.timeout)
            response.raise_for_status()
            parsed = parse_vllm_metrics(response.text, self.model_name)
            self._metrics_error = None
            return parsed, time.perf_counter() - started
        except Exception as exc:
            self._metrics_error = f"{type(exc).__name__}: {exc}"
            return None, time.perf_counter() - started

    def _prepare_nvml(self) -> None:
        if self.nvml_device is None:
            return
        try:
            import pynvml  # type: ignore[import-not-found]

            self._nvml = pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.nvml_device)
            self._nvml_error = None
        except Exception as exc:
            self._nvml_error = f"{type(exc).__name__}: {exc}"
            if self._nvml_initialized and self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass
                self._nvml_initialized = False

    async def prepare(self) -> dict[str, Any]:
        """Probe configured sources and return an idempotent compact receipt.

        A vLLM probe is available only when every required series is present
        and valid. Partial metrics remain visible through ``missing_metrics``
        and do not produce an affirmative availability result.
        """
        if self._prepared:
            assert self._probe_receipt is not None
            return self._copy_probe_receipt()
        if self._closed:
            raise RuntimeError("telemetry has been closed")
        self._prepare_nvml()
        nvml_probe = self._read_nvml()
        metrics, _ = await self._read_metrics()
        missing_metrics = list(metrics["missing_metrics"]) if metrics else []
        if metrics is not None:
            self._last_metrics = metrics
            self._last_counters = {
                field: metrics[field] for field in _COUNTER_FIELDS
            }
            self._last_counter_time = time.perf_counter()
        requested = {
            "vllm": self.metrics_url is not None,
            "nvml": self.nvml_device is not None,
        }
        self._probe_receipt = {
            "requested_sources": requested,
            "probe_errors": {
                "vllm": self._metrics_error,
                "nvml": self._nvml_error,
            },
            "missing_metrics": missing_metrics,
            "invalid_metrics": list(metrics["invalid_metrics"]) if metrics else [],
            "availability": {
                "vllm": (None if not requested["vllm"] else
                         bool(metrics is not None and not missing_metrics
                              and not metrics["invalid_metrics"])),
                "nvml": (None if not requested["nvml"] else
                         bool(nvml_probe["nvml_available"])),
            },
        }
        self._prepared = True
        return self._copy_probe_receipt()

    def _copy_probe_receipt(self) -> dict[str, Any]:
        assert self._probe_receipt is not None
        return copy.deepcopy(self._probe_receipt)

    def _read_nvml(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "nvml_source": "local_host" if self.nvml_device is not None else None,
            "nvml_device": self.nvml_device,
            "nvml_gpu_utilization_percent": None,
            "nvml_vram_used_bytes": None,
            "nvml_vram_total_bytes": None,
            "nvml_available": None if self.nvml_device is None else False,
        }
        if self.nvml_device is None or self._nvml_handle is None:
            return result
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            result.update({
                "nvml_gpu_utilization_percent": float(utilization.gpu),
                "nvml_vram_used_bytes": int(memory.used),
                "nvml_vram_total_bytes": int(memory.total),
            })
            if not math.isfinite(result["nvml_gpu_utilization_percent"]):
                result["nvml_gpu_utilization_percent"] = None
                raise ValueError("NVML returned a non-finite GPU utilization")
            result["nvml_available"] = True
            self._nvml_error = None
        except Exception as exc:
            self._nvml_error = f"{type(exc).__name__}: {exc}"
        return result

    async def collect(self) -> dict[str, Any]:
        """Collect one sample. Failed scrapes are explicit and never stale-filled."""
        if not self._prepared:
            await self.prepare()
        if self._closed:
            raise RuntimeError("telemetry has been closed")

        metrics, scrape_seconds = await self._read_metrics()
        now = time.perf_counter()
        values: dict[str, Any] = {
            "vllm_running_requests": None,
            "vllm_waiting_requests": None,
            "prompt_tokens_total": None,
            "generation_tokens_total": None,
            "prompt_tokens_per_second": None,
            "generation_tokens_per_second": None,
            "kv_cache_fraction": None,
            "missing_metrics": [],
            "invalid_metrics": [],
            "counter_resets": [],
            "metrics_scrape_seconds": scrape_seconds,
        }
        if metrics is not None:
            elapsed = (now - self._last_counter_time
                       if self._last_counter_time is not None else 0.0)
            rates, resets = counter_rates(metrics, self._last_counters, elapsed)
            values.update({
                "vllm_running_requests": metrics["running_requests"],
                "vllm_waiting_requests": metrics["waiting_requests"],
                "prompt_tokens_total": metrics["prompt_tokens_total"],
                "generation_tokens_total": metrics["generation_tokens_total"],
                "prompt_tokens_per_second": rates["prompt_tokens_total"],
                "generation_tokens_per_second": rates["generation_tokens_total"],
                "kv_cache_fraction": metrics["kv_cache_fraction"],
                "missing_metrics": metrics["missing_metrics"],
                "invalid_metrics": metrics["invalid_metrics"],
                "counter_resets": resets,
            })
            self._last_metrics = metrics
            self._last_counters = {
                field: metrics[field] for field in _COUNTER_FIELDS
            }
            self._last_counter_time = now
        values.update(self._read_nvml())
        requested = {
            "vllm": self.metrics_url is not None,
            "nvml": self.nvml_device is not None,
        }
        values["requested_sources"] = requested
        values["availability"] = {
            "vllm": (None if not requested["vllm"] else
                     bool(metrics is not None and not values["missing_metrics"]
                          and not values["invalid_metrics"])),
            "nvml": values["nvml_available"],
        }
        values["errors"] = {
            "metrics": self._metrics_error,
            "nvml": self._nvml_error,
        }
        return values

    async def sample_loop(self, recorder: Any, stop: asyncio.Event) -> None:
        """Emit ``resource_sample`` events until ``stop`` is set, without busy waits."""
        if not self._prepared:
            await self.prepare()
        loop_started = time.perf_counter()
        measurement_started = getattr(recorder, "start", loop_started)
        if not self._probe_emitted and self._probe_receipt is not None:
            recorder.emit("__resources__", "resource_probe", **self._copy_probe_receipt())
            self._probe_emitted = True
        while not stop.is_set():
            collection_started = time.perf_counter()
            sample = await self.collect()
            collection_finished = time.perf_counter()
            if stop.is_set():
                break
            collection_seconds = max(0.0, collection_finished - collection_started)
            sample["collection_started_offset_seconds"] = max(
                0.0, collection_started - measurement_started)
            sample["collection_finished_offset_seconds"] = max(
                0.0, collection_finished - measurement_started)
            sample["collection_seconds"] = collection_seconds
            recorder.emit("__resources__", "resource_sample", **sample)
            remaining = self.interval - (time.perf_counter() - collection_started)
            if remaining <= 0:
                await asyncio.sleep(0)
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        """Close the persistent HTTPX client and shut down this NVML session."""
        if self._closed:
            return
        try:
            if self._client is not None:
                await self._client.aclose()
        finally:
            if self._nvml_initialized and self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                finally:
                    self._nvml_initialized = False
            self._closed = True

    async def __aenter__(self) -> "Telemetry":
        await self.prepare()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
