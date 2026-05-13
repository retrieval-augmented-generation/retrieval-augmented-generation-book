# ch10.5-regression.py - Compare current ablation results to a saved baseline
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CURRENT = Path(__file__).parent / "ablation_results.json"
BASELINE = Path(__file__).parent / "ablation_baseline.json"
THRESHOLD = 0.05

METRICS = (
    "candidate_recall", "context_recall", "pool_mrr", "rerank_mrr",
    "faithfulness", "relevance", "citation_accuracy",
)


def main():
    if not CURRENT.exists():
        print(f"Run ch10.4 first to produce {CURRENT.name}")
        return
    current = {row["config"]: row for row in json.loads(CURRENT.read_text())}

    if not BASELINE.exists():
        BASELINE.write_text(CURRENT.read_text())
        print(f"No baseline found. Promoted {CURRENT.name} -> {BASELINE.name}.")
        print("The next run of ch10.4 will write a new current and this script "
              "will diff against this baseline.")
        return

    baseline = {row["config"]: row for row in json.loads(BASELINE.read_text())}

    print(f"Comparing current results to baseline. A drop of more than "
          f"{THRESHOLD:.2f} on any metric is flagged as a regression.\n")
    print(f"{'Config':<22}{'Metric':<18}{'Baseline':<10}{'Current':<10}{'Δ':<8}{'Status':<10}")
    print("-" * 78)

    regressions = 0
    for config in current:
        if config not in baseline:
            continue
        for metric in METRICS:
            b = baseline[config].get(metric)
            c = current[config].get(metric)
            if b is None or c is None:
                continue
            delta = c - b
            status = "ok"
            if delta < -THRESHOLD:
                status = "REGRESSION"
                regressions += 1
            elif delta > THRESHOLD:
                status = "improved"
            print(f"{config:<22}{metric:<18}{b:<10.3f}{c:<10.3f}{delta:<+8.3f}{status:<10}")

    print()
    print(f"Regressions detected: {regressions}")


if __name__ == "__main__":
    main()
