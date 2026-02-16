import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from feature_engineering import add_features
from scoring import probability_to_score, segment_and_decision


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
LR_PATH = os.path.join(ART_DIR, "logistic.pkl")
XGB_PATH = os.path.join(ART_DIR, "xgb.pkl")
BEST_MODEL_PATH = os.path.join(ART_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH = os.path.join(ART_DIR, "best_model_name.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_cols.pkl")
MODEL_VERSION_PATH = os.path.join(ART_DIR, "model_version.pkl")


def ensure_model_artifacts() -> None:
    missing = [p for p in [LR_PATH, XGB_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Train first with `python train_model.py`. "
            f"Missing: {missing}"
        )


@lru_cache(maxsize=1)
def load_artifacts():
    ensure_model_artifacts()
    lr = joblib.load(LR_PATH)
    xgb = joblib.load(XGB_PATH)
    best_model = None
    best_model_name = None
    if os.path.exists(BEST_MODEL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
    if os.path.exists(BEST_MODEL_NAME_PATH):
        best_model_name = str(joblib.load(BEST_MODEL_NAME_PATH))
    feature_cols = joblib.load(FEATURES_PATH)
    model_version = "v_unknown"
    if os.path.exists(MODEL_VERSION_PATH):
        model_version = str(joblib.load(MODEL_VERSION_PATH))
    return lr, xgb, best_model, best_model_name, feature_cols, model_version


def _build_model_input(application: Dict[str, Any], alternate: Dict[str, Any] | None) -> pd.DataFrame:
    alt = alternate or {}
    month_value = application.get("application_ts", "")[:10] or "January"

    monthly_salary = float(application.get("monthly_salary") or 0.0)
    emi = float(application.get("emi") or 0.0)
    invested = float(application.get("invested_amount") or 0.0)

    row = {
        "Customer_ID": application.get("customer_id"),
        "Month": month_value,
        "Age": application.get("age"),
        "Annual_Income": application.get("annual_income"),
        "Monthly_Inhand_Salary": application.get("monthly_salary"),
        "Num_of_Loan": application.get("num_of_loan"),
        "Interest_Rate": application.get("interest_rate"),
        "Delay_from_due_date": application.get("delay_days"),
        "Num_of_Delayed_Payment": application.get("delayed_payments"),
        "Outstanding_Debt": application.get("outstanding_debt"),
        "Credit_Utilization_Ratio": application.get("credit_utilization"),
        "Total_EMI_per_month": application.get("emi"),
        "Amount_invested_monthly": application.get("invested_amount"),
        "Monthly_Balance": monthly_salary - emi - invested,
        "Num_Credit_Inquiries": application.get("credit_inquiries"),
        "Occupation": application.get("occupation"),
        "Credit_Mix": application.get("credit_mix"),
        "Payment_Behaviour": application.get("payment_behaviour"),
        "Payment_of_Min_Amount": application.get("payment_of_min_amount"),
        "has_credit_history": application.get("has_credit_history", 1),
        "upi_txn_count_30d": alt.get("upi_txn_count_30d"),
        "utility_bill_ontime_ratio": alt.get("utility_bill_ontime_ratio"),
        "recharge_regularity_score": alt.get("recharge_regularity_score"),
        "spending_consistency_score": alt.get("spending_consistency_score"),
    }
    return pd.DataFrame([row])


def _compute_shap_factors(xgb_pipeline, x_row: pd.DataFrame, top_n: int = 5) -> Tuple[list[dict], str | None]:
    try:
        import shap

        prep = xgb_pipeline.named_steps["prep"]
        clf = xgb_pipeline.named_steps["clf"]

        transformed = prep.transform(x_row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[-1])
        else:
            shap_array = np.asarray(shap_values)

        values = shap_array[0]
        top_idx = np.argsort(np.abs(values))[::-1][:top_n]

        factors = []
        for idx in top_idx:
            clean_name = str(feature_names[idx]).replace("num__", "").replace("cat__", "")
            factors.append(
                {
                    "feature": clean_name,
                    "impact": float(values[idx]),
                    "direction": "Risk Up" if values[idx] > 0 else "Risk Down",
                }
            )
        return factors, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def predict_from_records(application: Dict[str, Any], alternate: Dict[str, Any] | None = None) -> Dict[str, Any]:
    lr, xgb, best_model, best_model_name, feature_cols, model_version = load_artifacts()
    raw = _build_model_input(application, alternate)
    feat = add_features(raw)
    x = feat[feature_cols]

    model_used = "lr_xgb_ensemble"
    if best_model is not None and best_model_name:
        pd_value = float(best_model.predict_proba(x)[:, 1][0])
        model_used = best_model_name
    else:
        pd_lr = float(lr.predict_proba(x)[:, 1][0])
        pd_xgb = float(xgb.predict_proba(x)[:, 1][0])
        pd_value = 0.35 * pd_lr + 0.65 * pd_xgb

    score = probability_to_score(pd_value)
    risk_level, decision = segment_and_decision(score)
    top_factors, explain_error = _compute_shap_factors(xgb, x, top_n=5)

    return {
        "probability_default": float(pd_value),
        "credit_score": int(score),
        "risk_level": risk_level,
        "decision": decision,
        "model_version": model_version,
        "scoring_model": model_used,
        "top_factors": top_factors,
        "explain_error": explain_error,
    }
