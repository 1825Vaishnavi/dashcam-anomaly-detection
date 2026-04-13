"""
monitoring/drift_detection.py
Evidently AI production drift detection.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def frames_to_feature_df(predictions):
    rows = []
    for p in predictions:
        rows.append({
            "confidence":        p.get("confidence", 0),
            "latency_ms":        p.get("latency_ms", 0),
            "is_anomaly":        int(p.get("is_anomaly", False)),
            "prob_normal":       p.get("class_probabilities", {}).get("normal", 0),
            "prob_accident":     p.get("class_probabilities", {}).get("accident", 0),
            "prob_obstacle":     p.get("class_probabilities", {}).get("obstacle", 0),
            "prob_pedestrian":   p.get("class_probabilities", {}).get("pedestrian", 0),
            "prob_traffic_sign": p.get("class_probabilities", {}).get("traffic_sign", 0),
            "prob_lane_viol":    p.get("class_probabilities", {}).get("lane_violation", 0),
        })
    return pd.DataFrame(rows)


def detect_drift(reference_predictions, current_predictions,
                 output_dir="monitoring/reports", threshold=0.15):
    ref_df = frames_to_feature_df(reference_predictions)
    cur_df = frames_to_feature_df(current_predictions)

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=ref_df, current_data=cur_df)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    report.save_html(html_path)

    result_dict = report.as_dict()
    drift_share = result_dict["metrics"][1]["result"].get(
        "share_of_drifted_columns", 0)
    dataset_drifted = drift_share > threshold

    summary = {
        "timestamp":       timestamp,
        "drift_share":     round(drift_share, 4),
        "dataset_drifted": dataset_drifted,
        "threshold":       threshold,
        "reference_size":  len(ref_df),
        "current_size":    len(cur_df),
        "report_path":     html_path,
    }

    if dataset_drifted:
        logger.warning(f"DRIFT DETECTED — {drift_share:.1%} drifted")
    else:
        logger.info(f"No drift — {drift_share:.1%} drifted")

    json_path = os.path.join(
        output_dir, f"drift_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def check_latency_sla(latencies_ms, sla_p99_ms=100.0):
    arr = np.array(latencies_ms)
    p99 = float(np.percentile(arr, 99))
    status = {
        "sla_passed":   p99 <= sla_p99_ms,
        "p99_ms":       round(p99, 2),
        "p95_ms":       round(float(np.percentile(arr, 95)), 2),
        "avg_ms":       round(float(np.mean(arr)), 2),
        "sla_limit_ms": sla_p99_ms,
        "sample_size":  len(latencies_ms),
    }
    if not status["sla_passed"]:
        logger.warning(
            f"SLA BREACH: p99={p99:.1f}ms > {sla_p99_ms}ms")
    return status