"""Offline paired-run summaries and one three-panel scientific figure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = {
    "completed_per_gpu_hour": "Completed trajectories / allocated GPU-hour",
    "model_starvation_seconds": "Sampled model starvation (seconds)",
    "cpu_core_seconds_per_completed_trajectory": "Client + tool CPU-core-seconds / completion",
}


def aggregate(comparison: dict) -> dict:
    if comparison.get("status") != "complete" or comparison.get("phase") != "comparison":
        raise ValueError("Only completed, unprofiled comparison receipts can be summarized")
    pairs = comparison.get("pairs", [])
    if not pairs:
        raise ValueError("No paired full runs")
    result = {"independent_unit": "paired full run", "pairs": len(pairs), "metrics": {},
              "quality": [], "claim": "Descriptive systems measurement; completion is not mathematical correctness."}
    seen = set()
    for pair in pairs:
        if pair["pair_id"] in seen:
            raise ValueError("Duplicate pair identifier")
        seen.add(pair["pair_id"])
        arms = pair["arms"]
        if set(arms) != {"fifo", "asyncroll"} or arms["fifo"]["seed"] != arms["asyncroll"]["seed"]:
            raise ValueError("Each pair requires FIFO and AsyncRoll with the same decoding seed")
        for policy, arm in arms.items():
            if arm.get("nvtx_enabled", False) or not arm["resource_metrics"].get("dedicated_single_gpu_declared"):
                raise ValueError("Profiled or non-dedicated runs cannot enter GPU-hour comparison")
            result["quality"].append({"pair_id": pair["pair_id"], "policy": policy,
                **{k: arm.get(k) for k in ("completed", "failed", "action_valid_rate", "trajectory_tool_call_rate")}})
    for key in METRICS:
        values = {p: [pair["arms"][p]["resource_metrics"].get(key) for pair in pairs]
                  for p in ("fifo", "asyncroll")}
        complete = all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                       for series in values.values() for v in series)
        result["metrics"][key] = {"values": values, "complete": complete,
            "median": {p: statistics.median(v) if complete else None for p, v in values.items()},
            "paired_asyncroll_over_fifo": [a / f if complete and f > 0 else None
                for f, a in zip(values["fifo"], values["asyncroll"])]}
    return result


def write_report(comparison_path: Path, output: Path, figure: bool = False) -> dict:
    data = aggregate(json.loads(comparison_path.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=False)
    (output / "aggregate.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = ["# FIFO versus AsyncRoll", "", data["claim"], "",
             f"Independent paired full runs: {data['pairs']}. Raw pairs are retained; no request-level confidence interval.", "",
             "| Metric | FIFO median | AsyncRoll median |", "| --- | ---: | ---: |"]
    for key, label in METRICS.items():
        medians = data["metrics"][key]["median"]
        cells = [f"{medians[p]:.6g}" if medians[p] is not None else "unknown" for p in ("fifo", "asyncroll")]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "CPU cost excludes the vLLM serving process. GPU-hours cover measurement, excluding startup/warmup. "
                  "Starvation is a sampled association; inspect coverage and Nsight before causal attribution.", "",
                  "## Quality and failures", "", "| Pair | Policy | Completed | Failed | Valid action rate | Tool-use rate |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for q in data["quality"]:
        lines.append("| " + " | ".join(str(q[k]) for k in ("pair_id", "policy", "completed", "failed", "action_valid_rate", "trajectory_tool_call_rate")) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, (key, label) in zip(axes, METRICS.items()):
            metric = data["metrics"][key]
            if metric["complete"]:
                for f, a in zip(metric["values"]["fifo"], metric["values"]["asyncroll"]):
                    ax.plot([0, 1], [f, a], "o-", color="#78909c", alpha=0.65, linewidth=1)
                ax.scatter([0, 1], [metric["median"][p] for p in ("fifo", "asyncroll")],
                           marker="D", s=60, color=["#1565c0", "#d84315"], zorder=3, label="Median")
                ax.set_ylim(bottom=0)
            else:
                ax.text(0.5, 0.5, "Incomplete measurement", ha="center", transform=ax.transAxes)
            ax.set_xticks([0, 1], ["FIFO", "AsyncRoll"])
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"Fixed resources; {data['pairs']} paired full runs (lines)")
        fig.savefig(output / "comparison.pdf")
        fig.savefig(output / "comparison.png", dpi=180)
        plt.close(fig)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", action="store_true", help="Requires asyncroll[figures]")
    args = parser.parse_args()
    write_report(args.comparison, args.output, args.figure)


if __name__ == "__main__":
    main()
