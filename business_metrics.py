from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from database import get_connection


def _rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def compute_business_metrics() -> Dict[str, Any]:
    conn = get_connection()

    predictions = pd.read_sql_query(
        """
        SELECT p.*
        FROM predictions p
        INNER JOIN (
            SELECT application_id, MAX(id) AS max_id
            FROM predictions
            GROUP BY application_id
        ) latest ON latest.max_id = p.id
        """,
        conn,
    )
    applications = pd.read_sql_query("SELECT * FROM applications", conn)
    outcomes = pd.read_sql_query("SELECT * FROM loan_outcomes", conn)
    alerts = pd.read_sql_query("SELECT * FROM alerts", conn)
    conn.close()

    metrics: Dict[str, Any] = {
        "applications": int(len(applications)),
        "predictions": int(len(predictions)),
        "outcomes": int(len(outcomes)),
        "alerts": int(len(alerts)),
    }

    if predictions.empty:
        metrics.update(
            {
                "approval_rate": 0.0,
                "model_npa_rate_on_approved": 0.0,
                "baseline_npa_rate_on_approved": 0.0,
                "npa_reduction_pct": 0.0,
                "thin_file_approval_rate_model": 0.0,
                "thin_file_approval_rate_baseline": 0.0,
                "thin_file_approval_uplift_pct_points": 0.0,
                "alert_hit_rate": 0.0,
                "portfolio_risk_distribution": {},
            }
        )
        return metrics

    pred_join = predictions.merge(
        applications[["id", "has_credit_history", "delay_days", "delayed_payments", "credit_utilization"]],
        left_on="application_id",
        right_on="id",
        how="left",
        suffixes=("", "_app"),
    ).merge(
        outcomes[["application_id", "default_flag"]],
        on="application_id",
        how="left",
    )

    pred_join["approved_model"] = pred_join["decision"].str.lower().ne("reject")
    metrics["approval_rate"] = _rate(pred_join["approved_model"])

    approved_with_outcomes = pred_join[pred_join["approved_model"] & pred_join["default_flag"].notna()]
    metrics["model_npa_rate_on_approved"] = _rate(approved_with_outcomes["default_flag"])

    baseline_approve = (
        (pred_join["has_credit_history"].fillna(0) == 1)
        & (pred_join["delay_days"].fillna(999) <= 15)
        & (pred_join["delayed_payments"].fillna(999) <= 6)
        & (pred_join["credit_utilization"].fillna(999) <= 45)
    )
    baseline_with_outcomes = pred_join[baseline_approve & pred_join["default_flag"].notna()]
    baseline_npa = _rate(baseline_with_outcomes["default_flag"])
    model_npa = float(metrics["model_npa_rate_on_approved"])
    metrics["baseline_npa_rate_on_approved"] = baseline_npa
    metrics["npa_reduction_pct"] = (
        float(((baseline_npa - model_npa) / baseline_npa) * 100.0) if baseline_npa > 0 else 0.0
    )

    thin_file = pred_join[pred_join["has_credit_history"].fillna(1) == 0].copy()
    if thin_file.empty:
        metrics["thin_file_approval_rate_model"] = 0.0
        metrics["thin_file_approval_rate_baseline"] = 0.0
        metrics["thin_file_approval_uplift_pct_points"] = 0.0
    else:
        thin_model_rate = _rate(thin_file["approved_model"])
        thin_baseline_rate = _rate(baseline_approve.loc[thin_file.index])
        metrics["thin_file_approval_rate_model"] = thin_model_rate
        metrics["thin_file_approval_rate_baseline"] = thin_baseline_rate
        metrics["thin_file_approval_uplift_pct_points"] = (thin_model_rate - thin_baseline_rate) * 100.0

    defaults = outcomes[outcomes["default_flag"] == 1].copy()
    if defaults.empty:
        metrics["alert_hit_rate"] = 0.0
    else:
        alerts["alert_ts"] = pd.to_datetime(alerts["alert_ts"], errors="coerce")
        defaults["observed_at"] = pd.to_datetime(defaults["observed_at"], errors="coerce")

        hit_count = 0
        for row in defaults.itertuples(index=False):
            prior = alerts[
                (alerts["customer_id"] == row.customer_id)
                & (alerts["alert_ts"].notna())
                & (alerts["alert_ts"] <= row.observed_at)
            ]
            if not prior.empty:
                hit_count += 1
        metrics["alert_hit_rate"] = hit_count / len(defaults)

    risk_dist = predictions["risk_level"].value_counts(normalize=True).to_dict()
    metrics["portfolio_risk_distribution"] = {k: float(v) for k, v in risk_dist.items()}
    metrics["approval_rate"] = float(metrics["approval_rate"])
    metrics["model_npa_rate_on_approved"] = float(metrics["model_npa_rate_on_approved"])
    metrics["baseline_npa_rate_on_approved"] = float(metrics["baseline_npa_rate_on_approved"])
    metrics["alert_hit_rate"] = float(metrics["alert_hit_rate"])
    return metrics


def metrics_frame_for_ui() -> pd.DataFrame:
    m = compute_business_metrics()
    rows = [
        ("Applications", str(m["applications"])),
        ("Predictions", str(m["predictions"])),
        ("Outcomes", str(m["outcomes"])),
        ("Alerts", str(m["alerts"])),
        ("Approval Rate", f"{m['approval_rate'] * 100:.2f}%"),
        ("Model NPA Rate (Approved)", f"{m['model_npa_rate_on_approved'] * 100:.2f}%"),
        ("Baseline NPA Rate (Approved)", f"{m['baseline_npa_rate_on_approved'] * 100:.2f}%"),
        ("NPA Reduction", f"{m['npa_reduction_pct']:.2f}%"),
        ("Thin-File Approval Uplift", f"{m['thin_file_approval_uplift_pct_points']:.2f} pp"),
        ("Alert Hit Rate", f"{m['alert_hit_rate'] * 100:.2f}%"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
