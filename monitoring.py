from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

from behavior_monitor import early_default_alert
from persistence import get_latest_application_id, get_transactions, insert_alert, list_customer_ids


SPEND_TYPES = {"DEBIT", "SPEND", "PURCHASE", "BILL", "RECHARGE", "UPI"}


def _pct_change(previous: float, current: float) -> float:
    if previous is None or np.isnan(previous) or previous == 0:
        return 0.0
    return ((current - previous) / abs(previous)) * 100.0


def compute_behavior_snapshot(customer_id: str, lookback_days: int = 120) -> Dict[str, Any]:
    tx = get_transactions(customer_id, lookback_days=lookback_days)
    if tx.empty:
        return {
            "customer_id": customer_id,
            "tx_count": 0,
            "emi_delay_trend": 0,
            "balance_drop_pct": 0.0,
            "spending_spike_pct": 0.0,
            "alert": False,
            "risk_level": "Low",
            "reasons": [],
        }

    tx["txn_ts"] = pd.to_datetime(tx["txn_ts"], errors="coerce")
    tx = tx.dropna(subset=["txn_ts"]).sort_values("txn_ts")
    now = tx["txn_ts"].max()

    last_30_start = now - timedelta(days=30)
    prev_30_start = now - timedelta(days=60)
    last_7_start = now - timedelta(days=7)
    prev_7_start = now - timedelta(days=14)

    tx["txn_type_norm"] = tx["txn_type"].astype(str).str.upper()
    spend_tx = tx[tx["txn_type_norm"].isin(SPEND_TYPES)].copy()
    spend_tx["amount"] = pd.to_numeric(spend_tx["amount"], errors="coerce").fillna(0.0)
    current_spend = spend_tx.loc[spend_tx["txn_ts"] >= last_30_start, "amount"].sum()
    previous_spend = spend_tx.loc[
        (spend_tx["txn_ts"] >= prev_30_start) & (spend_tx["txn_ts"] < last_30_start),
        "amount",
    ].sum()
    spending_spike_pct = max(_pct_change(previous_spend, current_spend), 0.0)

    balance = tx[["txn_ts", "balance_after"]].copy()
    balance["balance_after"] = pd.to_numeric(balance["balance_after"], errors="coerce")
    current_balance = balance.loc[balance["txn_ts"] >= last_7_start, "balance_after"].mean()
    previous_balance = balance.loc[
        (balance["txn_ts"] >= prev_7_start) & (balance["txn_ts"] < last_7_start),
        "balance_after",
    ].mean()
    balance_drop_pct = max(-_pct_change(previous_balance, current_balance), 0.0)

    emi_tx = tx[tx["is_emi"] == 1].copy()
    emi_delay_trend = 0
    if not emi_tx.empty:
        emi_tx["emi_due_date"] = pd.to_datetime(emi_tx["emi_due_date"], errors="coerce")
        emi_tx["delay_days"] = (
            (emi_tx["txn_ts"] - emi_tx["emi_due_date"]).dt.days.clip(lower=0).fillna(0)
        )
        delays = emi_tx.sort_values("txn_ts")["delay_days"].tolist()
        emi_delay_trend = sum(1 for i in range(1, len(delays)) if delays[i] > delays[i - 1])

    alert = early_default_alert(
        emi_delay_trend=emi_delay_trend,
        balance_drop_pct=float(balance_drop_pct),
        spending_spike_pct=float(spending_spike_pct),
    )

    return {
        "customer_id": customer_id,
        "tx_count": int(len(tx)),
        "emi_delay_trend": int(emi_delay_trend),
        "balance_drop_pct": float(round(balance_drop_pct, 2)),
        "spending_spike_pct": float(round(spending_spike_pct, 2)),
        "alert": bool(alert["alert"]),
        "risk_level": alert["risk_level"],
        "reasons": alert["reasons"],
    }


def monitor_customer(customer_id: str, application_id: int | None = None) -> Dict[str, Any]:
    snapshot = compute_behavior_snapshot(customer_id)
    if snapshot["alert"]:
        target_app_id = application_id if application_id is not None else get_latest_application_id(customer_id)
        alert_id = insert_alert(
            {
                "customer_id": customer_id,
                "application_id": target_app_id,
                "alert_type": "EARLY_DEFAULT_WARNING",
                "risk_level": snapshot["risk_level"],
                "reasons": "; ".join(snapshot["reasons"]),
                "status": "open",
            }
        )
        snapshot["alert_id"] = alert_id
    return snapshot


def monitor_all_customers(limit: int | None = None) -> Dict[str, Any]:
    customer_ids = list_customer_ids(limit=limit)
    results = []
    alerts_generated = 0

    for cid in customer_ids:
        snap = monitor_customer(cid)
        if snap.get("alert"):
            alerts_generated += 1
        results.append(snap)

    return {
        "customers_checked": len(customer_ids),
        "alerts_generated": alerts_generated,
        "results": results,
    }
