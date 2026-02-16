import numpy as np
import pandas as pd


MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")


def _ensure_col(df: pd.DataFrame, col: str, default=np.nan) -> None:
    if col not in df.columns:
        df[col] = default


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Num_Credit_Inquiries",
        "Payment_Behaviour",
        "Month",
        "Customer_ID",
        "has_credit_history",
        "upi_txn_count_30d",
        "utility_bill_ontime_ratio",
        "recharge_regularity_score",
        "spending_consistency_score",
    ]
    for c in required:
        _ensure_col(out, c)

    num_cols = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Num_Credit_Inquiries",
        "has_credit_history",
        "upi_txn_count_30d",
        "utility_bill_ontime_ratio",
        "recharge_regularity_score",
        "spending_consistency_score",
    ]
    for c in num_cols:
        out[c] = _to_numeric(out[c])

    salary = out["Monthly_Inhand_Salary"].replace(0, np.nan)
    income = out["Annual_Income"].replace(0, np.nan)

    out["debt_to_income"] = out["Outstanding_Debt"] / income
    out["loan_burden_ratio"] = out["Total_EMI_per_month"] / salary
    out["payment_consistency_score"] = (1 - (out["Num_of_Delayed_Payment"] / 30.0)).clip(0, 1)
    out["spending_volatility"] = (out["Amount_invested_monthly"] / salary).clip(lower=0)
    out["credit_utilization"] = out["Credit_Utilization_Ratio"] / 100.0

    pb = out["Payment_Behaviour"].astype(str)
    out["alt_payment_behavior_score"] = np.where(pb.str.contains("Low_spent"), 0.8, 0.4)
    out["alt_payment_behavior_score"] += np.where(pb.str.contains("Small_value"), 0.1, 0.0)
    out["alt_payment_behavior_score"] = out["alt_payment_behavior_score"].clip(0, 1)

    month_text = out["Month"].astype(str)
    month_map = month_text.map(MONTH_MAP)
    month_from_date = pd.to_datetime(month_text, errors="coerce").dt.month
    out["month_num"] = month_map.fillna(month_from_date).fillna(0)

    out["utility_bill_ontime_ratio"] = out["utility_bill_ontime_ratio"].clip(0, 1)
    out["recharge_regularity_score"] = out["recharge_regularity_score"].clip(0, 1)
    out["spending_consistency_score"] = out["spending_consistency_score"].clip(0, 1)
    out["upi_txn_count_30d"] = out["upi_txn_count_30d"].clip(lower=0)

    out["alt_data_score"] = (
        0.35 * (out["upi_txn_count_30d"] / 100.0).clip(0, 1)
        + 0.30 * out["utility_bill_ontime_ratio"].fillna(0.5)
        + 0.20 * out["recharge_regularity_score"].fillna(0.5)
        + 0.15 * out["spending_consistency_score"].fillna(0.5)
    ).clip(0, 1)

    out = out.sort_values(["Customer_ID", "month_num"]).copy()
    out["balance_drop"] = (
        out.groupby("Customer_ID")["Monthly_Balance"].diff().fillna(0).clip(upper=0).abs()
    )

    out["behavior_risk_signal"] = (
        (out["Delay_from_due_date"].fillna(0) > 20).astype(int)
        + (out["balance_drop"].fillna(0) > 500).astype(int)
        + (out["credit_utilization"].fillna(0) > 0.7).astype(int)
    )

    return out
