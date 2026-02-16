import os
from typing import Any, Dict

import requests

from persistence import get_transactions


def _safe_get(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int = 8) -> dict:
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected response type from {url}")
    return data


def _fallback_from_transactions(customer_id: str) -> Dict[str, Any]:
    tx = get_transactions(customer_id, lookback_days=30)
    if tx.empty:
        return {
            "upi_txn_count_30d": 0,
            "utility_bill_ontime_ratio": 0.5,
            "recharge_regularity_score": 0.5,
            "spending_consistency_score": 0.5,
            "source": "transaction_fallback",
        }

    tx_type = tx["txn_type"].astype(str).str.upper()
    upi_count = int((tx_type == "UPI").sum())
    bill_count = int((tx_type == "BILL").sum())
    recharge_count = int((tx_type == "RECHARGE").sum())
    spend_count = int(tx_type.isin(["UPI", "DEBIT", "SPEND", "PURCHASE", "BILL", "RECHARGE"]).sum())

    utility_ratio = 1.0 if bill_count > 0 else 0.6
    recharge_score = min(recharge_count / 5.0, 1.0) if recharge_count > 0 else 0.4
    spend_consistency = min(spend_count / 20.0, 1.0)

    return {
        "upi_txn_count_30d": upi_count,
        "utility_bill_ontime_ratio": round(float(utility_ratio), 3),
        "recharge_regularity_score": round(float(recharge_score), 3),
        "spending_consistency_score": round(float(spend_consistency), 3),
        "source": "transaction_fallback",
    }


def fetch_alternate_data(customer_id: str, allow_fallback: bool = True) -> Dict[str, Any]:
    upi_url = os.getenv("UPI_API_URL", "").strip()
    utility_url = os.getenv("UTILITY_API_URL", "").strip()
    recharge_url = os.getenv("RECHARGE_API_URL", "").strip()
    api_key = os.getenv("ALT_DATA_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    out: Dict[str, Any] = {
        "upi_txn_count_30d": None,
        "utility_bill_ontime_ratio": None,
        "recharge_regularity_score": None,
        "spending_consistency_score": None,
        "source": "external_feed",
    }

    errors = []
    try:
        if upi_url:
            data = _safe_get(upi_url, headers=headers, params={"customer_id": customer_id})
            out["upi_txn_count_30d"] = int(data.get("upi_txn_count_30d", 0))
            if "spending_consistency_score" in data:
                out["spending_consistency_score"] = float(data["spending_consistency_score"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"UPI feed error: {exc}")

    try:
        if utility_url:
            data = _safe_get(utility_url, headers=headers, params={"customer_id": customer_id})
            out["utility_bill_ontime_ratio"] = float(data.get("utility_bill_ontime_ratio", 0.5))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Utility feed error: {exc}")

    try:
        if recharge_url:
            data = _safe_get(recharge_url, headers=headers, params={"customer_id": customer_id})
            out["recharge_regularity_score"] = float(data.get("recharge_regularity_score", 0.5))
            if out["spending_consistency_score"] is None and "spending_consistency_score" in data:
                out["spending_consistency_score"] = float(data["spending_consistency_score"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Recharge feed error: {exc}")

    complete = all(out.get(k) is not None for k in [
        "upi_txn_count_30d",
        "utility_bill_ontime_ratio",
        "recharge_regularity_score",
        "spending_consistency_score",
    ])

    if not complete and allow_fallback:
        fallback = _fallback_from_transactions(customer_id)
        for k in ["upi_txn_count_30d", "utility_bill_ontime_ratio", "recharge_regularity_score", "spending_consistency_score"]:
            if out.get(k) is None:
                out[k] = fallback[k]
        out["source"] = f"{out['source']}+fallback"
        complete = True

    if errors:
        out["integration_errors"] = errors
    out["complete"] = complete
    return out
