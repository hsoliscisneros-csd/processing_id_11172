"""Track C — reporte OOD: gap temporal vs espacial (PhysAgg A0–A3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_pr(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("metrics", {}).get("pr_auc")


def main() -> int:
    parser = argparse.ArgumentParser(description="OOD gap report Track C")
    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=Path("experiments/outputs/physagg"),
    )
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=Path("experiments/outputs/physagg/spatial"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/manifests/spatial_ood_v0.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/physagg/ood_gap_report.json"),
    )
    args = parser.parse_args()

    runs = ["A0", "A1", "A2", "A3"]
    rows = []
    for run in runs:
        pr_t = _load_pr(args.temporal_dir / f"metrics_{run}.json")
        pr_s = _load_pr(args.spatial_dir / f"metrics_{run}.json")
        if pr_t is None or pr_s is None:
            rows.append({"run": run, "pr_temporal": pr_t, "pr_spatial": pr_s, "delta": None})
            continue
        delta = pr_t - pr_s
        rows.append(
            {
                "run": run,
                "pr_temporal": pr_t,
                "pr_spatial": pr_s,
                "delta": delta,
                "delta_rel_to_temporal": delta / pr_t if pr_t else None,
            }
        )

    a0 = next((r for r in rows if r["run"] == "A0"), None)
    gate = {"passed": False, "detail": []}
    if a0 and a0.get("delta") is not None:
        for r in rows:
            if r["run"] == "A0" or r.get("delta") is None:
                continue
            # reduce gap ≥10% relativo vs A0
            reduction = (a0["delta"] - r["delta"]) / a0["delta"] if a0["delta"] else 0.0
            ok = reduction >= 0.10
            gate["detail"].append(
                {
                    "run": r["run"],
                    "gap_reduction_vs_A0": reduction,
                    "pass_10pct": ok,
                    "wins_spatial_vs_A0": (r["pr_spatial"] or 0) > (a0["pr_spatial"] or 0),
                }
            )
        gate["passed"] = any(d["pass_10pct"] for d in gate["detail"])

    report = {
        "track": "C",
        "manifest": str(args.manifest),
        "protocol": "P1_muni_holdout (ya corrido en physagg spatial)",
        "rows": rows,
        "gate_H_C": gate,
        "recommendation": (
            "A1/A2 reducen gap vs A0 si pass_10pct; "
            "si no, Track C = evaluación OOD del paper, no aporte de adaptación."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
