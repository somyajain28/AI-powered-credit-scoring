import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "lending.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    finally:
        cur.close()
        conn.close()


def init_db() -> None:
    with db_cursor() as cur:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                applicant_name TEXT NOT NULL,
                application_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                age INTEGER,
                occupation TEXT,
                credit_mix TEXT,
                has_credit_history INTEGER NOT NULL DEFAULT 1,
                annual_income REAL,
                monthly_salary REAL,
                num_of_loan INTEGER,
                interest_rate REAL,
                emi REAL,
                delayed_payments INTEGER,
                delay_days INTEGER,
                credit_utilization REAL,
                outstanding_debt REAL,
                invested_amount REAL,
                credit_inquiries INTEGER,
                payment_behaviour TEXT,
                payment_of_min_amount TEXT
            );

            CREATE TABLE IF NOT EXISTS alternate_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL DEFAULT CURRENT_DATE,
                upi_txn_count_30d INTEGER,
                utility_bill_ontime_ratio REAL,
                recharge_regularity_score REAL,
                spending_consistency_score REAL,
                source TEXT NOT NULL DEFAULT 'manual',
                UNIQUE(customer_id, as_of_date)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                customer_id TEXT NOT NULL,
                prediction_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                probability_default REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                decision TEXT NOT NULL,
                model_version TEXT NOT NULL,
                FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                txn_ts TEXT NOT NULL,
                txn_type TEXT NOT NULL,
                amount REAL NOT NULL,
                balance_after REAL,
                is_emi INTEGER NOT NULL DEFAULT 0,
                emi_due_date TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS loan_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                customer_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                days_past_due INTEGER NOT NULL DEFAULT 0,
                default_flag INTEGER NOT NULL CHECK(default_flag IN (0, 1)),
                FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                application_id INTEGER,
                alert_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                reasons TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS bank_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                customer_id TEXT,
                source_system TEXT NOT NULL DEFAULT 'bank_webhook',
                payload_json TEXT NOT NULL
            );
            """
        )
