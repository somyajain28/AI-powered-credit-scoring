import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from database import init_db
from persistence import (
    insert_application,
    insert_outcome,
    insert_transaction,
    upsert_alternate_data,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_DEFAULT = os.path.join(BASE_DIR, "loan_data.csv")

MONTH_TO_NUM = {
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


def _num(value):
    s = str(value)
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _month_to_date(month_name: str) -> str:
    m = MONTH_TO_NUM.get(str(month_name), 1)
    return datetime(2025, int(m), 1).date().isoformat()


def _default_flag(row: pd.Series) -> tuple[int, int]:
    delay_days = _num(row.get("Delay_from_due_date"))
    delayed_payments = _num(row.get("Num_of_Delayed_Payment"))
    min_pay = str(row.get("Payment_of_Min_Amount", "")).strip().upper()

    days_past_due = int(max(delay_days if not np.isnan(delay_days) else 0, 0))
    default = int(
        (days_past_due >= 45)
        or ((delayed_payments if not np.isnan(delayed_payments) else 0) >= 20)
        or (min_pay == "NO" and days_past_due >= 25)
    )
    return default, days_past_due


def _alt_payload(row: pd.Series) -> dict:
    delays = _num(row.get("Num_of_Delayed_Payment"))
    delays = 0 if np.isnan(delays) else delays
    pay_beh = str(row.get("Payment_Behaviour", ""))
    bank_accts = _num(row.get("Num_Bank_Accounts"))
    bank_accts = 3 if np.isnan(bank_accts) else bank_accts

    upi = int(max(5, bank_accts * 8 + random.randint(0, 35)))
    utility_ratio = float(np.clip(1.0 - (delays / 60.0), 0.2, 1.0))
    recharge_reg = 0.85 if "Small_value" in pay_beh else 0.65
    spend_consistency = 0.75 if "Low_spent" in pay_beh else 0.55

    return {
        "upi_txn_count_30d": upi,
        "utility_bill_ontime_ratio": round(utility_ratio, 3),
        "recharge_regularity_score": round(recharge_reg, 3),
        "spending_consistency_score": round(spend_consistency, 3),
    }


def _insert_transactions(customer_id: str, app_date: datetime, emi: float, delayed_payments: float) -> None:
    # Two spend events
    for delta in [45, 18]:
        ts = (app_date - timedelta(days=delta)).replace(hour=10)
        spend = round(random.uniform(500, 4500), 2)
        balance = round(random.uniform(1000, 90000), 2)
        insert_transaction(
            {
                "customer_id": customer_id,
                "txn_ts": ts.isoformat(sep=" ", timespec="seconds"),
                "txn_type": "UPI",
                "amount": spend,
                "balance_after": balance,
                "is_emi": 0,
            }
        )

    # EMI payment event with delay trend signal
    due = (app_date - timedelta(days=30)).date()
    delayed_payments = 0.0 if delayed_payments is None or np.isnan(delayed_payments) else delayed_payments
    delay_days = int(min(max(delayed_payments / 2, 0), 25))
    paid_ts = datetime.combine(due + timedelta(days=delay_days), datetime.min.time()) + timedelta(hours=11)
    insert_transaction(
        {
            "customer_id": customer_id,
            "txn_ts": paid_ts.isoformat(sep=" ", timespec="seconds"),
            "txn_type": "EMI_PAYMENT",
            "amount": round(float(emi) if not np.isnan(emi) else random.uniform(1000, 8000), 2),
            "balance_after": round(random.uniform(800, 70000), 2),
            "is_emi": 1,
            "emi_due_date": due.isoformat(),
        }
    )


def bootstrap(csv_path: str, rows: int) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    init_db()
    df = pd.read_csv(csv_path)
    if rows > 0 and rows < len(df):
        df = df.sample(rows, random_state=42).reset_index(drop=True)

    inserted = 0
    for _, row in df.iterrows():
        customer_id = str(row.get("Customer_ID", "")).strip()
        if not customer_id:
            continue

        app_date = datetime.fromisoformat(_month_to_date(row.get("Month")))
        credit_mix = str(row.get("Credit_Mix", "")).strip()
        has_credit_history = int(credit_mix not in {"", "_", "nan", "NaN"})

        payload = {
            "customer_id": customer_id,
            "applicant_name": str(row.get("Name", "Unknown Applicant")),
            "age": _num(row.get("Age")),
            "occupation": str(row.get("Occupation", "Unknown")),
            "credit_mix": credit_mix if credit_mix not in {"_", "nan", "NaN"} else None,
            "has_credit_history": has_credit_history,
            "annual_income": _num(row.get("Annual_Income")),
            "monthly_salary": _num(row.get("Monthly_Inhand_Salary")),
            "num_of_loan": _num(row.get("Num_of_Loan")),
            "interest_rate": _num(row.get("Interest_Rate")),
            "emi": _num(row.get("Total_EMI_per_month")),
            "delayed_payments": _num(row.get("Num_of_Delayed_Payment")),
            "delay_days": _num(row.get("Delay_from_due_date")),
            "credit_utilization": _num(row.get("Credit_Utilization_Ratio")),
            "outstanding_debt": _num(row.get("Outstanding_Debt")),
            "invested_amount": _num(row.get("Amount_invested_monthly")),
            "credit_inquiries": _num(row.get("Num_Credit_Inquiries")),
            "payment_behaviour": str(row.get("Payment_Behaviour", "Unknown")),
            "payment_of_min_amount": str(row.get("Payment_of_Min_Amount", "NM")),
        }
        app_id = insert_application(payload)

        alt = _alt_payload(row)
        upsert_alternate_data(
            {
                "customer_id": customer_id,
                "as_of_date": app_date.date().isoformat(),
                "source": "bootstrap_from_csv",
                **alt,
            }
        )

        default_flag, dpd = _default_flag(row)
        insert_outcome(
            {
                "application_id": app_id,
                "customer_id": customer_id,
                "observed_at": (app_date + timedelta(days=90)).date().isoformat(),
                "days_past_due": dpd,
                "default_flag": default_flag,
            }
        )

        _insert_transactions(
            customer_id=customer_id,
            app_date=app_date,
            emi=payload["emi"] if payload["emi"] is not None else 0,
            delayed_payments=payload["delayed_payments"] if payload["delayed_payments"] is not None else 0,
        )
        inserted += 1

    print(f"Bootstrap complete. Inserted applications/outcomes: {inserted}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap lending.db from loan_data.csv for end-to-end demo.")
    parser.add_argument("--csv-path", type=str, default=CSV_DEFAULT)
    parser.add_argument("--rows", type=int, default=8000, help="Sample rows to ingest (0 = all)")
    args = parser.parse_args()
    bootstrap(args.csv_path, args.rows)


if __name__ == "__main__":
    main()
