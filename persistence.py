import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from database import db_cursor, get_connection


def _row_to_dict(row) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


def insert_application(payload: Dict[str, Any]) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO applications (
                customer_id, applicant_name, age, occupation, credit_mix, has_credit_history,
                annual_income, monthly_salary, num_of_loan, interest_rate, emi,
                delayed_payments, delay_days, credit_utilization, outstanding_debt,
                invested_amount, credit_inquiries, payment_behaviour, payment_of_min_amount
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["customer_id"],
                payload["applicant_name"],
                payload.get("age"),
                payload.get("occupation"),
                payload.get("credit_mix"),
                int(payload.get("has_credit_history", 1)),
                payload.get("annual_income"),
                payload.get("monthly_salary"),
                payload.get("num_of_loan"),
                payload.get("interest_rate"),
                payload.get("emi"),
                payload.get("delayed_payments"),
                payload.get("delay_days"),
                payload.get("credit_utilization"),
                payload.get("outstanding_debt"),
                payload.get("invested_amount"),
                payload.get("credit_inquiries"),
                payload.get("payment_behaviour"),
                payload.get("payment_of_min_amount"),
            ),
        )
        return int(cur.lastrowid)


def upsert_alternate_data(payload: Dict[str, Any]) -> None:
    as_of_date = payload.get("as_of_date") or datetime.utcnow().date().isoformat()
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO alternate_data (
                customer_id, as_of_date, upi_txn_count_30d, utility_bill_ontime_ratio,
                recharge_regularity_score, spending_consistency_score, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(customer_id, as_of_date) DO UPDATE SET
                upi_txn_count_30d=excluded.upi_txn_count_30d,
                utility_bill_ontime_ratio=excluded.utility_bill_ontime_ratio,
                recharge_regularity_score=excluded.recharge_regularity_score,
                spending_consistency_score=excluded.spending_consistency_score,
                source=excluded.source
            """,
            (
                payload["customer_id"],
                as_of_date,
                payload.get("upi_txn_count_30d"),
                payload.get("utility_bill_ontime_ratio"),
                payload.get("recharge_regularity_score"),
                payload.get("spending_consistency_score"),
                payload.get("source", "manual"),
            ),
        )


def get_application(application_id: int) -> Dict[str, Any]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM applications WHERE id = ?", (application_id,))
        return _row_to_dict(cur.fetchone())


def get_latest_alternate_data(customer_id: str) -> Dict[str, Any]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM alternate_data
            WHERE customer_id = ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (customer_id,),
        )
        return _row_to_dict(cur.fetchone())


def get_application_with_alternate(application_id: int) -> Dict[str, Any]:
    app = get_application(application_id)
    if not app:
        return {}
    alt = get_latest_alternate_data(app["customer_id"])
    merged = dict(app)
    merged.update(
        {
            "upi_txn_count_30d": alt.get("upi_txn_count_30d"),
            "utility_bill_ontime_ratio": alt.get("utility_bill_ontime_ratio"),
            "recharge_regularity_score": alt.get("recharge_regularity_score"),
            "spending_consistency_score": alt.get("spending_consistency_score"),
        }
    )
    return merged


def insert_prediction(payload: Dict[str, Any]) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (
                application_id, customer_id, probability_default, credit_score,
                risk_level, decision, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["application_id"],
                payload["customer_id"],
                payload["probability_default"],
                payload["credit_score"],
                payload["risk_level"],
                payload["decision"],
                payload.get("model_version", "v1"),
            ),
        )
        return int(cur.lastrowid)


def list_latest_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY prediction_ts DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def insert_transaction(payload: Dict[str, Any]) -> int:
    metadata_json = payload.get("metadata_json")
    if isinstance(metadata_json, dict):
        metadata_json = json.dumps(metadata_json)

    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (
                customer_id, txn_ts, txn_type, amount, balance_after, is_emi, emi_due_date, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["customer_id"],
                payload["txn_ts"],
                payload["txn_type"],
                payload["amount"],
                payload.get("balance_after"),
                int(payload.get("is_emi", 0)),
                payload.get("emi_due_date"),
                metadata_json,
            ),
        )
        return int(cur.lastrowid)


def get_transactions(customer_id: str, lookback_days: int = 120) -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT *
        FROM transactions
        WHERE customer_id = ?
          AND datetime(txn_ts) >= datetime('now', ?)
        ORDER BY datetime(txn_ts) ASC
    """
    df = pd.read_sql_query(query, conn, params=(customer_id, f"-{lookback_days} day"))
    conn.close()
    return df


def insert_alert(payload: Dict[str, Any]) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO alerts (
                customer_id, application_id, alert_type, risk_level, reasons, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["customer_id"],
                payload.get("application_id"),
                payload["alert_type"],
                payload["risk_level"],
                payload["reasons"],
                payload.get("status", "open"),
            ),
        )
        return int(cur.lastrowid)


def list_open_alerts(limit: int = 100) -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM alerts
            WHERE status = 'open'
            ORDER BY alert_ts DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def insert_outcome(payload: Dict[str, Any]) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO loan_outcomes (
                application_id, customer_id, observed_at, days_past_due, default_flag
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload["application_id"],
                payload["customer_id"],
                payload["observed_at"],
                int(payload.get("days_past_due", 0)),
                int(payload["default_flag"]),
            ),
        )
        return int(cur.lastrowid)


def get_training_frame() -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT
            a.id AS application_id,
            a.customer_id AS Customer_ID,
            date(a.application_ts) AS Month,
            a.age AS Age,
            a.annual_income AS Annual_Income,
            a.monthly_salary AS Monthly_Inhand_Salary,
            a.num_of_loan AS Num_of_Loan,
            a.interest_rate AS Interest_Rate,
            a.delay_days AS Delay_from_due_date,
            a.delayed_payments AS Num_of_Delayed_Payment,
            a.outstanding_debt AS Outstanding_Debt,
            a.credit_utilization AS Credit_Utilization_Ratio,
            a.emi AS Total_EMI_per_month,
            a.invested_amount AS Amount_invested_monthly,
            (a.monthly_salary - a.emi - a.invested_amount) AS Monthly_Balance,
            a.credit_inquiries AS Num_Credit_Inquiries,
            a.occupation AS Occupation,
            a.credit_mix AS Credit_Mix,
            a.payment_behaviour AS Payment_Behaviour,
            a.payment_of_min_amount AS Payment_of_Min_Amount,
            a.has_credit_history AS has_credit_history,
            ad.upi_txn_count_30d AS upi_txn_count_30d,
            ad.utility_bill_ontime_ratio AS utility_bill_ontime_ratio,
            ad.recharge_regularity_score AS recharge_regularity_score,
            ad.spending_consistency_score AS spending_consistency_score,
            o.default_flag AS default_flag
        FROM applications a
        INNER JOIN loan_outcomes o
            ON o.application_id = a.id
        LEFT JOIN alternate_data ad
            ON ad.customer_id = a.customer_id
            AND ad.as_of_date = (
                SELECT MAX(ad2.as_of_date)
                FROM alternate_data ad2
                WHERE ad2.customer_id = a.customer_id
                  AND ad2.as_of_date <= date(a.application_ts)
            )
        ORDER BY a.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_customer_prediction_history(customer_id: str, limit: int = 50) -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT *
        FROM predictions
        WHERE customer_id = ?
        ORDER BY prediction_ts DESC, id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(customer_id, limit))
    conn.close()
    return df


def list_customer_ids(limit: int | None = None) -> List[str]:
    with db_cursor() as cur:
        if limit is None:
            cur.execute(
                """
                SELECT DISTINCT customer_id
                FROM applications
                ORDER BY customer_id
                """
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT customer_id
                FROM applications
                ORDER BY customer_id
                LIMIT ?
                """,
                (limit,),
            )
        return [r["customer_id"] for r in cur.fetchall()]


def get_latest_application_id(customer_id: str) -> Optional[int]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM applications
            WHERE customer_id = ?
            ORDER BY datetime(application_ts) DESC, id DESC
            LIMIT 1
            """,
            (customer_id,),
        )
        row = cur.fetchone()
        return int(row["id"]) if row is not None else None


def insert_bank_event(payload: Dict[str, Any]) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO bank_events (event_type, customer_id, source_system, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                payload["event_type"],
                payload.get("customer_id"),
                payload.get("source_system", "bank_webhook"),
                payload["payload_json"],
            ),
        )
        return int(cur.lastrowid)


def list_bank_events(limit: int = 100) -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM bank_events
            ORDER BY event_ts DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
