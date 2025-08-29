#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Thesis Results Summarizer
======================================

Collects and analyzes experimental results across all methods and log types
for comprehensive thesis evaluation. Generates detailed tables and statistics
required for academic publication.

Scans for:
- Transformer results: results/<log_type>/predictions.pkl, performance_metrics.json
- ML baselines: results/multilabel_*/summary.json
- XGBoost OvR: results/xgboost_ml/*/metrics_summary.json  
- Binary baselines: results/binary_*/metrics.json

Outputs:
- thesis_summary.json: Complete structured data
- thesis_summary.md: Human-readable summary
- thesis_tables.json: Formatted tables for LaTeX
- performance_analysis.json: Detailed performance breakdown
"""

import os
import sys
from pathlib import Path
import json
import re
import pickle
import numpy as np
from collections import defaultdict
import time
from datetime import datetime

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)


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


def detect_available_log_types():
    """Detect all available log types from results directories."""
    log_types = set()
    
    # From transformer results
    for child in RESULTS.iterdir():
        if child.is_dir() and (child / "predictions.pkl").exists():
            log_types.add(child.name)
    
    # From binary results
    for child in RESULTS.glob("binary_*"):
        if child.is_dir():
            log_type = child.name.replace("binary_", "")
            log_types.add(log_type)
    
    return sorted(list(log_types))

def read_enhanced_transformer_results(log_type: str):
    """Read comprehensive transformer results."""
    out = {"method": "transformer", "log_type": log_type, "available": False}
    
    pred_file = RESULTS / log_type / "predictions.pkl"
    perf_file = RESULTS / log_type / "performance_metrics.json"
    
    if pred_file.exists():
        try:
            with open(pred_file, "rb") as f:
                data = pickle.load(f)
            
            out["available"] = True
            
            # Extract performance metrics
            if "performance_metrics" in data:
                perf = data["performance_metrics"]
                out.update({
                    "dataset_size": perf.get("dataset_info", {}).get("test_samples", 0),
                    "training_time_minutes": perf.get("training_metrics", {}).get("total_training_time_minutes", 0),
                    "device": perf.get("training_metrics", {}).get("device", "unknown"),
                    "embedding_type": perf.get("embedding_type", "unknown")
                })
                
                if "evaluation_metrics" in perf:
                    out["metrics"] = perf["evaluation_metrics"]
                
        except Exception as e:
            print(f"Warning: Error reading {log_type}: {e}")
    
    return out

def generate_thesis_tables(all_results, dataset_chars):
    """Generate all thesis tables."""
    tables = {}
    
    # Table II: Performance Comparison
    perf_table = []
    for result in all_results:
        if result["available"] and "metrics" in result:
            metrics = result["metrics"]
            perf_table.append({
                "log_type": result["log_type"],
                "method": result.get("model_name", result["method"]),
                "micro_f1": metrics.get("micro_f1", 0),
                "macro_f1": metrics.get("macro_f1", 0),
                "hamming_loss": metrics.get("hamming_loss", 0),
                "jaccard_index": metrics.get("jaccard_index", 0),
                "training_time_minutes": result.get("training_time_minutes", 0)
            })
    tables["performance_comparison"] = perf_table
    
    return tables

def main():
    print("🔍 Comprehensive Thesis Results Analysis")
    print("="*50)
    
    # Detect available log types
    log_types = detect_available_log_types()
    print(f"Found results for {len(log_types)} log types: {', '.join(log_types)}")
    
    # Collect all results
    all_results = []
    transformer_results = {}
    binary_results = {}
    
    # Read transformer results
    for log_type in log_types:
        result = read_enhanced_transformer_results(log_type)
        transformer_results[log_type] = result
        all_results.append(result)
        
        # Read binary results
        bin_result = read_binary(log_type)
        if bin_result:
            binary_results[log_type] = bin_result
    
    # Read other results
    xgboost_results = read_xgb()
    ml_results = read_ml_models()
    
    # Calculate dataset characteristics
    dataset_chars = {}
    for log_type in log_types:
        if transformer_results[log_type]["available"]:
            result = transformer_results[log_type]
            dataset_chars[log_type] = {
                "total_entries": result.get("dataset_size", 0),
                "size_category": "Small" if result.get("dataset_size", 0) < 1000 else "Large",
                "num_classes": result.get("num_classes", 0)
            }
    
    # Generate comprehensive summary
    comprehensive_summary = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_log_types": len(log_types),
            "total_experiments": len(all_results)
        },
        "dataset_characteristics": dataset_chars,
        "transformer_results": transformer_results,
        "binary_results": binary_results,
        "xgboost_results": xgboost_results,
        "ml_results": ml_results,
        "thesis_tables": generate_thesis_tables(all_results, dataset_chars)
    }
    
    # Save results
    with open(RESULTS / "thesis_summary.json", "w") as f:
        json.dump(comprehensive_summary, f, indent=2)
    
    with open(RESULTS / "thesis_tables.json", "w") as f:
        json.dump(comprehensive_summary["thesis_tables"], f, indent=2)
    
    # Generate enhanced markdown
    lines = [
        "# LBAD Framework - Experimental Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Log Types:** {len(log_types)}",
        f"**Total Experiments:** {len(all_results)}",
        "",
        "## Dataset Characteristics",
        "",
        "| Log Type | Total Entries | Size Category | Classes |",
        "|----------|---------------|---------------|---------|"
    ]
    
    for log_type, chars in dataset_chars.items():
        lines.append(f"| {log_type} | {chars['total_entries']:,} | {chars['size_category']} | {chars['num_classes']} |")
    
    lines.extend([
        "",
        "## Performance Summary",
        ""
    ])
    
    for log_type, result in transformer_results.items():
        if result["available"] and "metrics" in result:
            metrics = result["metrics"]
            lines.append(f"### {log_type}")
            lines.append(f"- Micro-F1: {metrics.get('micro_f1', 0):.4f}")
            lines.append(f"- Macro-F1: {metrics.get('macro_f1', 0):.4f}")
            lines.append(f"- Training Time: {result.get('training_time_minutes', 0):.2f} minutes")
            lines.append("")
    
    with open(RESULTS / "thesis_summary.md", "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Thesis analysis complete:")
    print(f"   📊 {RESULTS / 'thesis_summary.json'}")
    print(f"   📋 {RESULTS / 'thesis_tables.json'}")
    print(f"   📝 {RESULTS / 'thesis_summary.md'}")


if __name__ == "__main__":
    main()