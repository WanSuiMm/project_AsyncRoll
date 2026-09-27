"""Write a Chrome trace and dependency-free HTML resource timeline.

The timeline separates trajectory, model-client, and CPU queue/worker spans.
Model bars represent HTTP/client request time and must not be read as GPU
execution. Resource samples show vLLM server counters and, when enabled, local
NVML host utilization/memory; utilization is not attributed to a request.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


_EVENT_FIELDS = {"t", "id", "event"}


def _time(event: dict[str, Any]) -> float | None:
    try:
        value = float(event["t"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _identity(event: dict[str, Any], specific: str) -> tuple[Any, ...]:
    # Keeping ID/turn/kind supports logs without request_id/job_id while still
    # distinguishing concurrent requests when those IDs are present.
    return (
        event.get("id"), event.get(specific), event.get("turn"), event.get("kind"),
    )


def _args(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key not in _EVENT_FIELDS}


def _label(event: dict[str, Any], base: str) -> str:
    bits = [base]
    if event.get("id") is not None:
        bits.append(str(event["id"]))
    if event.get("turn") is not None:
        bits.append(f"turn {event['turn']}")
    if event.get("tool_name"):
        bits.append(str(event["tool_name"]))
    elif event.get("kind"):
        bits.append(str(event["kind"]))
    return " · ".join(bits)


def _entity_track(base: str, event: dict[str, Any]) -> str:
    entity = event.get("id")
    return f"{base} / {entity if entity is not None else 'unknown'}"


def _finite_number(event: dict[str, Any], key: str) -> float | None:
    try:
        value = float(event[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _paired_spans(
    events: list[dict[str, Any]],
    start_name: str,
    end_names: tuple[str, ...],
    *,
    identity_field: str,
    track_for_start: Any,
    base_label: str,
    duration: float,
) -> list[dict[str, Any]]:
    pending: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
    spans: list[dict[str, Any]] = []
    for event in events:
        event_time = _time(event)
        if event_time is None:
            continue
        name = event.get("event")
        key = _identity(event, identity_field)
        if name == start_name:
            pending[key].append(event)
        elif name in end_names:
            if pending[key]:
                start = pending[key].popleft()
                start_time = _time(start)
                assert start_time is not None
                end_time = max(start_time, event_time)
                args = _args(start)
                args.update(_args(event))
                outcome = "failed" if name.endswith("failed") else "finished"
                args["outcome"] = outcome
                spans.append({
                    "track": track_for_start(start),
                    "label": _label(start, base_label),
                    "start": start_time,
                    "end": end_time,
                    "args": args,
                })
            else:
                # An isolated completion stays visible as a zero-length marker.
                spans.append({
                    "track": track_for_start(event),
                    "label": _label(event, f"{base_label} unmatched completion"),
                    "start": event_time,
                    "end": event_time,
                    "args": {**_args(event), "outcome": "unmatched completion"},
                })

    for starts in pending.values():
        while starts:
            start = starts.popleft()
            start_time = _time(start)
            assert start_time is not None
            end_time = max(start_time, duration)
            spans.append({
                "track": track_for_start(start),
                "label": _label(start, f"{base_label} incomplete"),
                "start": start_time,
                "end": end_time,
                "args": {**_args(start), "outcome": "incomplete"},
            })
    return spans


def _trajectory_spans(events: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    pending: dict[Any, deque[dict[str, Any]]] = defaultdict(deque)
    spans: list[dict[str, Any]] = []
    for event in events:
        event_time = _time(event)
        if event_time is None:
            continue
        name = event.get("event")
        identity = event.get("id")
        if name == "trajectory_started":
            pending[identity].append(event)
        elif name == "trajectory_finished":
            if pending[identity]:
                start = pending[identity].popleft()
                start_time = _time(start)
                assert start_time is not None
                spans.append({
                    "track": _entity_track("Trajectory", start),
                    "label": _label(start, "Trajectory"),
                    "start": start_time,
                    "end": max(start_time, event_time),
                    "args": {**_args(start), **_args(event), "outcome": event.get("status", "finished")},
                })
            else:
                spans.append({
                    "track": _entity_track("Trajectory", event),
                    "label": _label(event, "Trajectory unmatched completion"),
                    "start": event_time,
                    "end": event_time,
                    "args": {**_args(event), "outcome": "unmatched completion"},
                })
    for starts in pending.values():
        while starts:
            start = starts.popleft()
            start_time = _time(start)
            assert start_time is not None
            spans.append({
                "track": _entity_track("Trajectory", start),
                "label": _label(start, "Trajectory incomplete"),
                "start": start_time,
                "end": max(start_time, duration),
                "args": {**_args(start), "outcome": "incomplete"},
            })
    return spans


def _cpu_child_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create exact child-code intervals from worker timestamps and durations."""
    spans: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") not in {"cpu_finished", "cpu_failed"}:
            continue
        started = _finite_number(event, "worker_started")
        execute_seconds = _finite_number(event, "execute_seconds")
        if started is None or execute_seconds is None or execute_seconds < 0:
            continue
        finished = _finite_number(event, "worker_finished")
        worker_id = event.get("worker_id", "unknown")
        outcome = "failed" if event["event"] == "cpu_failed" else "finished"
        args = _args(event)
        args["outcome"] = outcome
        spans.append({
            "track": f"CPU child execution / worker {worker_id}",
            "label": _label(event, "CPU child execution"),
            "start": started,
            "end": started + execute_seconds,
            "args": args,
        })
        # Keep the independently measured child wall endpoint in the trace args
        # so serialization or teardown time can be inspected separately.
        if finished is not None:
            spans[-1]["args"]["worker_finished_offset_seconds"] = finished
    return spans


def _resource_samples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "resource_sample":
            continue
        event_time = _time(event)
        if event_time is None:
            continue
        samples.append({"t": event_time, **_args(event)})
    return samples


def _source_status(
    events: list[dict[str, Any]], samples: list[dict[str, Any]],
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    probe_event = next((event for event in events
                        if event.get("event") == "resource_probe"), None)
    probe_data = probe if probe is not None else (_args(probe_event) if probe_event else None)
    latest = samples[-1] if samples else None
    return {
        "probe": probe_data,
        "latest_sample": ({
            "requested_sources": latest.get("requested_sources"),
            "availability": latest.get("availability"),
            "errors": latest.get("errors"),
            "missing_metrics": latest.get("missing_metrics"),
            "invalid_metrics": latest.get("invalid_metrics"),
        } if latest is not None else None),
    }


def build_timeline(
    events: Iterable[dict[str, Any]], probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert Recorder events into spans, resource samples, and Chrome events."""
    ordered = sorted((dict(event) for event in events), key=lambda item: _time(item) or 0.0)
    timestamps = [value for event in ordered if (value := _time(event)) is not None]
    duration = max(timestamps, default=0.0)

    spans = _trajectory_spans(ordered, duration)
    span_specs = (
        ("model_enqueued", ("model_started",), "request_id",
         lambda event: _entity_track("Model queue", event), "Model client queue"),
        ("model_started", ("model_finished", "model_failed"), "request_id",
         lambda event: _entity_track("Model HTTP client request", event), "HTTP client request"),
        ("cpu_enqueued", ("cpu_started",), "job_id",
         lambda event: _entity_track("CPU queue", event), "CPU queue"),
        ("cpu_started", ("cpu_finished", "cpu_failed"), "job_id",
         lambda event: (f"CPU dispatch / worker {event['worker_id']}"
                        if event.get("worker_id") is not None
                        else f"CPU dispatch / {event.get('id', 'unknown')}"), "CPU dispatch"),
        ("gpu_enqueued", ("gpu_started",), "request_id",
         lambda event: _entity_track("Backend queue", event), "Backend queue"),
        ("gpu_started", ("gpu_finished", "gpu_failed"), "request_id",
         lambda event: _entity_track("Backend request (client)", event), "Backend client call"),
    )
    for start_name, end_names, identity_field, track, label in span_specs:
        spans.extend(_paired_spans(
            ordered, start_name, end_names, identity_field=identity_field,
            track_for_start=track, base_label=label, duration=duration,
        ))
    spans.extend(_cpu_child_spans(ordered))
    spans.sort(key=lambda span: (span["start"], span["track"], span["label"]))
    samples = _resource_samples(ordered)
    source_status = _source_status(ordered, samples, probe)

    trace_events: list[dict[str, Any]] = [
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "AsyncRoll client and resource samples"}}
    ]
    tracks = list(dict.fromkeys(span["track"] for span in spans))
    thread_ids = {track: index + 1 for index, track in enumerate(tracks)}
    for track, thread_id in thread_ids.items():
        trace_events.append({
            "name": "thread_name", "ph": "M", "pid": 1, "tid": thread_id,
            "args": {"name": track},
        })
    for span in spans:
        start = span["start"]
        trace_events.append({
            "name": span["label"], "cat": span["track"], "ph": "X",
            "pid": 1, "tid": thread_ids[span["track"]],
            "ts": round(start * 1_000_000),
            "dur": round(max(0.0, span["end"] - start) * 1_000_000),
            "args": span["args"],
        })
    for sample in samples:
        trace_events.append({
            "name": "vLLM and local NVML resource sample", "cat": "Resources",
            "ph": "i", "s": "g", "pid": 1,
            "ts": round(sample["t"] * 1_000_000), "args": sample,
        })
    trace_events.sort(key=lambda event: event.get("ts", -1))
    return {
        "duration_seconds": duration,
        "spans": spans,
        "samples": samples,
        "source_status": source_status,
        "trace_events": trace_events,
    }


def _standalone_html(timeline: dict[str, Any], source_events: list[dict[str, Any]]) -> str:
    embedded = json.dumps(
        {"timeline": timeline, "events": source_events},
        ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False,
    )
    # Escape HTML script delimiters and JavaScript line separators.
    embedded = (embedded.replace("&", "\\u0026").replace("<", "\\u003c")
                .replace(">", "\\u003e").replace("\u2028", "\\u2028")
                .replace("\u2029", "\\u2029"))
    title = "AsyncRoll resource timeline"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#17212b;background:#fff}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:#536273;margin:4px 0 14px}}
.frame{{overflow-x:auto;border:1px solid #d7dee7;border-radius:8px}}
canvas{{display:block;min-width:760px;width:100%;height:auto}}
details{{margin-top:14px}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;border-radius:6px}}
#source-status{{font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#fff7df;border:1px solid #ead89e;padding:10px;border-radius:6px;margin:12px 0}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Model bars show HTTP/client request time. CPU dispatch spans include worker IPC; child execution is timed separately. vLLM series come from the server metrics endpoint; NVML values describe the local host. GPU utilization is not assigned to individual requests.</p>
<pre id="source-status" aria-label="Telemetry source status"></pre>
<div class="frame"><canvas id="chart"></canvas></div>
<details><summary>Raw events and sample values</summary><pre id="raw"></pre></details>
<script>
(function(){{
const DATA = {embedded};
document.getElementById("raw").textContent = JSON.stringify(DATA.events, null, 2);
const sourceStatus = DATA.timeline.source_status;
document.getElementById("source-status").textContent =
  sourceStatus.probe || sourceStatus.latest_sample
    ? JSON.stringify(sourceStatus, null, 2)
    : "No telemetry probe or resource samples were recorded.";
const spans = DATA.timeline.spans;
const samples = DATA.timeline.samples;
const colors = {{finished:"#4b89dc",failed:"#d94d4d",incomplete:"#7e8793", "unmatched completion":"#a573c8"}};
const tracks = [...new Set(spans.map(s => s.track))];
const metricDefs = [
  ["vLLM running requests", "vllm_running_requests", v => String(v)],
  ["vLLM waiting requests", "vllm_waiting_requests", v => String(v)],
  ["Prompt tokens / second", "prompt_tokens_per_second", v => Number(v).toFixed(1)],
  ["Generation tokens / second", "generation_tokens_per_second", v => Number(v).toFixed(1)],
  ["vLLM KV cache fraction", "kv_cache_fraction", v => Number(v).toFixed(3)],
  ["Local NVML GPU utilization (%)", "nvml_gpu_utilization_percent", v => Number(v).toFixed(0) + "%"],
  ["Local NVML VRAM (used / total bytes)", "nvml_vram_used_bytes", v => Number(v).toLocaleString()]
];
const resources = metricDefs.filter(([,key]) => samples.some(s => s[key] != null));
const rows = [...tracks.map(name => ({{kind:"span",name}})), ...resources.map(([name,key,format]) => ({{kind:"metric",name,key,format}}))];
const W = Math.max(760, document.getElementById("chart").clientWidth || 760);
const left = 270, right = 26, plotTop = 30, rowH = 34;
const H = plotTop + rows.length * rowH + 20;
const canvas = document.getElementById("chart"), dpr = window.devicePixelRatio || 1;
canvas.width = W*dpr; canvas.height = H*dpr; canvas.style.height = H+"px";
const ctx = canvas.getContext("2d"); ctx.scale(dpr,dpr);
const end = Math.max(DATA.timeline.duration_seconds, 0.001);
const plotW = W-left-right, x = t => left + (t/end)*plotW;
ctx.fillStyle="#fff"; ctx.fillRect(0,0,W,H);
ctx.font="11px system-ui,sans-serif"; ctx.textBaseline="middle";
for(let i=0;i<=5;i++){{const t=end*i/5, xx=x(t),label=end<1?(t*1000).toPrecision(3)+" ms":t.toFixed(2)+" s";ctx.strokeStyle="#e7ebef";ctx.beginPath();ctx.moveTo(xx,plotTop-12);ctx.lineTo(xx,H-10);ctx.stroke();ctx.fillStyle="#637282";ctx.textAlign="center";ctx.fillText(label,xx,12)}}
rows.forEach((row,index)=>{{
  const y=plotTop+index*rowH;
  ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(0,y+rowH-1);ctx.lineTo(W,y+rowH-1);ctx.stroke();
  ctx.fillStyle="#253443";ctx.textAlign="left";ctx.fillText(row.name,10,y+rowH/2);
  if(row.kind==="span"){{
    for(const s of spans.filter(v=>v.track===row.name)){{
      const xx=x(s.start), ww=Math.max(2,x(s.end)-xx), outcome=s.args.outcome||"finished";
      ctx.globalAlpha=outcome==="incomplete"?0.65:0.9;ctx.fillStyle=colors[outcome]||colors.finished;
      ctx.fillRect(xx,y+8,ww,rowH-16);ctx.globalAlpha=1;
      if(ww>55){{ctx.fillStyle="#fff";ctx.textAlign="left";ctx.fillText(s.label.slice(0,Math.max(8,Math.floor(ww/6))),xx+4,y+rowH/2)}}
    }}
  }} else {{
    const usable=samples.map(s=>s[row.key]).filter(v=>v!=null&&Number.isFinite(Number(v))).map(Number);
    const max=Math.max(...usable,1e-12), pts=samples.filter(s=>s[row.key]!=null&&Number.isFinite(Number(s[row.key])));
    ctx.strokeStyle="#258578";ctx.fillStyle="#258578";ctx.lineWidth=1.5;ctx.beginPath();
    let active=false;
    for(const s of samples){{const raw=s[row.key],value=Number(raw);if(raw==null||!Number.isFinite(value)){{active=false;continue}}const xx=x(s.t),yy=y+rowH-7-(value/max)*(rowH-17);if(active)ctx.lineTo(xx,yy);else ctx.moveTo(xx,yy);active=true}}
    ctx.stroke();
    pts.forEach(s=>{{const xx=x(s.t),yy=y+rowH-7-(Number(s[row.key])/max)*(rowH-17);ctx.beginPath();ctx.arc(xx,yy,2.5,0,Math.PI*2);ctx.fill()}});
    if(pts.length){{const last=pts[pts.length-1],value=row.key==="nvml_vram_used_bytes"?`${{row.format(last[row.key])}} / ${{Number(last.nvml_vram_total_bytes||0).toLocaleString()}}`:row.format(last[row.key]);ctx.fillStyle="#536273";ctx.textAlign="right";ctx.fillText("latest: "+value,W-8,y+rowH/2)}}
  }}
}});
}})();
</script></body></html>'''


def write_timeline(
    events: Iterable[dict[str, Any]], output: Path,
    probe: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``trace.json`` and self-contained ``timeline.html`` under output.

    Returns ``(trace_path, html_path)``. ``output`` is a directory, normally a
    run directory; source events are embedded in HTML for portable inspection.
    """
    source_events = [dict(event) for event in events]
    timeline = build_timeline(source_events, probe=probe)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.json"
    html_path = output / "timeline.html"
    trace_path.write_text(json.dumps({
        "traceEvents": timeline["trace_events"],
        "displayTimeUnit": "ms",
        "metadata": {
            "description": "Client request, CPU queue/dispatch/child execution, trajectory and resource samples",
            "model_span_semantics": "HTTP/client time; not GPU execution",
            "nvml_semantics": "Local host telemetry; not attributed to requests",
            "duration_seconds": timeline["duration_seconds"],
        },
    }, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    html_path.write_text(_standalone_html(timeline, source_events), encoding="utf-8")
    return trace_path, html_path
