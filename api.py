import json
import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from business_metrics import compute_business_metrics
from database import init_db
from integrations import fetch_alternate_data
from monitoring import monitor_all_customers, monitor_customer
from persistence import (
    get_application_with_alternate,
    get_customer_prediction_history,
    insert_application,
    insert_bank_event,
    insert_outcome,
    insert_prediction,
    insert_transaction,
    list_bank_events,
    list_latest_predictions,
    list_open_alerts,
    upsert_alternate_data,
)
from predictor import ensure_model_artifacts, predict_from_records


app = FastAPI(title="AI Lending Risk Intelligence API", version="1.1.0")


def _require_bank_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("BANK_API_KEY", "dev-bank-key")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid bank API key")


class AlternateDataInput(BaseModel):
    as_of_date: Optional[str] = None
    upi_txn_count_30d: int = Field(ge=0, default=0)
    utility_bill_ontime_ratio: float = Field(ge=0, le=1, default=0.5)
    recharge_regularity_score: float = Field(ge=0, le=1, default=0.5)
    spending_consistency_score: float = Field(ge=0, le=1, default=0.5)
    source: str = "api"


class ApplicationInput(BaseModel):
    customer_id: str
    applicant_name: str
    age: int
    occupation: str
    credit_mix: Optional[str] = None
    has_credit_history: int = 1
    annual_income: float
    monthly_salary: float
    num_of_loan: int
    interest_rate: float
    emi: float
    delayed_payments: int
    delay_days: int
    credit_utilization: float
    outstanding_debt: float
    invested_amount: float
    credit_inquiries: int
    payment_behaviour: str
    payment_of_min_amount: str
    alternate_data: Optional[AlternateDataInput] = None


class BankApplicationEvent(BaseModel):
    event_id: Optional[str] = None
    source_system: str = "bank_webhook"
    application: ApplicationInput
    auto_score: bool = True


class TransactionInput(BaseModel):
    customer_id: str
    txn_ts: Optional[str] = None
    txn_type: str
    amount: float
    balance_after: Optional[float] = None
    is_emi: int = 0
    emi_due_date: Optional[str] = None


class BankTransactionBatch(BaseModel):
    event_id: Optional[str] = None
    source_system: str = "bank_webhook"
    transactions: List[TransactionInput]
    auto_monitor: bool = True


class OutcomeInput(BaseModel):
    application_id: int
    customer_id: str
    observed_at: Optional[str] = None
    days_past_due: int = 0
    default_flag: int = Field(ge=0, le=1)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.post("/applications")
def create_application(payload: ApplicationInput):
    app_id = insert_application(payload.model_dump(exclude={"alternate_data"}))
    if payload.alternate_data:
        upsert_alternate_data(
            {
                "customer_id": payload.customer_id,
                **payload.alternate_data.model_dump(),
            }
        )
    return {"application_id": app_id}


@app.post("/predict/{application_id}")
def score_application(application_id: int):
    ensure_model_artifacts()
    record = get_application_with_alternate(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found")

    result = predict_from_records(record, record)
    prediction_id = insert_prediction(
        {
            "application_id": application_id,
            "customer_id": record["customer_id"],
            "probability_default": result["probability_default"],
            "credit_score": result["credit_score"],
            "risk_level": result["risk_level"],
            "decision": result["decision"],
            "model_version": result["model_version"],
        }
    )
    return {"prediction_id": prediction_id, **result}


@app.post("/transactions")
def ingest_transaction(payload: TransactionInput):
    tx_payload = payload.model_dump()
    tx_payload["txn_ts"] = tx_payload["txn_ts"] or datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    tx_id = insert_transaction(tx_payload)
    snapshot = monitor_customer(payload.customer_id)
    return {"transaction_id": tx_id, "monitoring": snapshot}


@app.post("/outcomes")
def add_outcome(payload: OutcomeInput):
    out_payload = payload.model_dump()
    out_payload["observed_at"] = out_payload["observed_at"] or datetime.utcnow().date().isoformat()
    outcome_id = insert_outcome(out_payload)
    return {"outcome_id": outcome_id}


@app.get("/monitor/{customer_id}")
def monitor(customer_id: str):
    return monitor_customer(customer_id)


@app.post("/monitor/run-all")
def run_portfolio_monitoring(limit: int = 0):
    summary = monitor_all_customers(limit=None if limit <= 0 else limit)
    return summary


@app.get("/metrics/business")
def business_metrics():
    return compute_business_metrics()


@app.get("/dashboard/summary")
def dashboard_summary():
    return {
        "latest_predictions": list_latest_predictions(limit=100),
        "open_alerts": list_open_alerts(limit=100),
        "bank_events": list_bank_events(limit=100),
        "business_metrics": compute_business_metrics(),
    }


@app.get("/customers/{customer_id}/history")
def customer_history(customer_id: str):
    df = get_customer_prediction_history(customer_id, limit=100)
    return {"customer_id": customer_id, "history": df.to_dict(orient="records")}


@app.post("/integrations/alternate-data/pull/{customer_id}")
def pull_alternate_data(
    customer_id: str,
    allow_fallback: bool = True,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_bank_key(x_api_key)
    data = fetch_alternate_data(customer_id, allow_fallback=allow_fallback)
    payload = {
        "customer_id": customer_id,
        "as_of_date": datetime.utcnow().date().isoformat(),
        "upi_txn_count_30d": data["upi_txn_count_30d"],
        "utility_bill_ontime_ratio": data["utility_bill_ontime_ratio"],
        "recharge_regularity_score": data["recharge_regularity_score"],
        "spending_consistency_score": data["spending_consistency_score"],
        "source": data.get("source", "external_feed"),
    }
    upsert_alternate_data(payload)
    insert_bank_event(
        {
            "event_type": "alternate_data_pull",
            "customer_id": customer_id,
            "source_system": "external_feed",
            "payload_json": json.dumps(data),
        }
    )
    return data


@app.post("/integrations/bank/webhook/application")
def bank_application_webhook(
    payload: BankApplicationEvent,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_bank_key(x_api_key)

    app_dict = payload.application.model_dump(exclude={"alternate_data"})
    app_id = insert_application(app_dict)

    alt = payload.application.alternate_data.model_dump() if payload.application.alternate_data else None
    if alt:
        upsert_alternate_data(
            {
                "customer_id": payload.application.customer_id,
                **alt,
            }
        )

    event_id = insert_bank_event(
        {
            "event_type": "application_webhook",
            "customer_id": payload.application.customer_id,
            "source_system": payload.source_system,
            "payload_json": payload.model_dump_json(),
        }
    )

    out = {"application_id": app_id, "bank_event_id": event_id}
    if payload.auto_score:
        ensure_model_artifacts()
        record = get_application_with_alternate(app_id)
        pred = predict_from_records(record, record)
        pred_id = insert_prediction(
            {
                "application_id": app_id,
                "customer_id": record["customer_id"],
                "probability_default": pred["probability_default"],
                "credit_score": pred["credit_score"],
                "risk_level": pred["risk_level"],
                "decision": pred["decision"],
                "model_version": pred["model_version"],
            }
        )
        out["prediction_id"] = pred_id
        out["prediction"] = pred
    return out


@app.post("/integrations/bank/webhook/transactions")
def bank_transactions_webhook(
    payload: BankTransactionBatch,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_bank_key(x_api_key)
    inserted = 0
    customer_ids = set()

    for tx in payload.transactions:
        tx_payload = tx.model_dump()
        tx_payload["txn_ts"] = tx_payload["txn_ts"] or datetime.utcnow().isoformat(sep=" ", timespec="seconds")
        insert_transaction(tx_payload)
        inserted += 1
        customer_ids.add(tx.customer_id)

    event_id = insert_bank_event(
        {
            "event_type": "transactions_webhook",
            "customer_id": None,
            "source_system": payload.source_system,
            "payload_json": payload.model_dump_json(),
        }
    )

    monitoring_results = []
    if payload.auto_monitor:
        for cid in sorted(customer_ids):
            monitoring_results.append(monitor_customer(cid))

    return {
        "bank_event_id": event_id,
        "transactions_inserted": inserted,
        "customers_touched": len(customer_ids),
        "monitoring_results": monitoring_results,
    }
