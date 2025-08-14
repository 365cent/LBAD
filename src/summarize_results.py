#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize experiment results across embeddings and methods.

- Scans results/ for:
  - results/<log_type>/predictions.pkl and enhanced_evaluation_report.txt (unsupervised transformer)
  - results/xgboost_ml/*/metrics_summary.json
  - results/multilabel_* (ml_models.py reports)
  - results/binary_<log_type>/metrics.json
- Produces results/thesis_summary.json and results/thesis_summary.md
"""

from pathlib import Path
import json
import re
import pickle

RESULTS = Path("results")


def read_transformer(log_type: str):
    out = {}
    pred_file = RESULTS / log_type / "predictions.pkl"
    if pred_file.exists():
        try:
            with open(pred_file, "rb") as f:
                data = pickle.load(f)
            metrics_file = RESULTS / log_type / "enhanced_evaluation_report.txt"
            out["predictions"] = True
            if metrics_file.exists():
                out["report"] = str(metrics_file)
        except Exception:
            pass
    return out


def read_binary(log_type: str):
    f = RESULTS / f"binary_{log_type}" / "metrics.json"
    if f.exists():
        try:
            with open(f, "r") as h:
                return json.load(h)
        except Exception:
            return None
    return None


def read_xgb():
    out = []
    for p in (RESULTS / "xgboost_ml").glob("**/metrics_summary.json"):
        try:
            with open(p, "r") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def read_ml_models():
    # Parse ml_models summary jsons if present under results/multilabel_<log_type>_*/summary.json
    out = []
    for p in RESULTS.glob("multilabel_*_*/summary.json"):
        try:
            with open(p, "r") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def main():
    summary = {
        "transformer": {},
        "binary": {},
        "xgboost_ml": read_xgb(),
        "ml_models": read_ml_models(),
    }

    # Detect log types from transformer outputs
    for child in RESULTS.iterdir():
        if child.is_dir() and (child / "predictions.pkl").exists():
            log_type = child.name
            summary["transformer"][log_type] = read_transformer(log_type)
            summary["binary"][log_type] = read_binary(log_type)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "thesis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Lightweight MD
    lines = ["# Thesis Experiment Summary", ""]
    for lt, tr in summary["transformer"].items():
        lines.append(f"## {lt}")
        lines.append(f"- Transformer outputs: {'yes' if tr else 'no'}")
        bin_m = summary["binary"].get(lt)
        if bin_m and "results" in bin_m:
            lines.append("- Binary baselines:")
            for model, m in bin_m["results"].items():
                lines.append(f"  - {model.upper()}: F1 {m.get('f1', 0):.3f} Acc {m.get('accuracy',0):.3f}")
        lines.append("")
    with open(RESULTS / "thesis_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {RESULTS / 'thesis_summary.json'} and {RESULTS / 'thesis_summary.md'}")


if __name__ == "__main__":
    main()