import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from business_metrics import compute_business_metrics, metrics_frame_for_ui
from database import init_db
from monitoring import monitor_all_customers, monitor_customer
from persistence import (
    get_application,
    insert_application,
    insert_outcome,
    insert_prediction,
    insert_transaction,
    list_latest_predictions,
    list_open_alerts,
    upsert_alternate_data,
)
from predictor import ensure_model_artifacts, predict_from_records


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "applicant"


def build_report_text(result: dict) -> str:
    lines = [
        "AI Credit Risk Report",
        "=====================",
        f"Application ID: {result['application_id']}",
        f"Prediction ID: {result['prediction_id']}",
        f"Applicant: {result['name']}",
        f"Customer ID: {result['customer_id']}",
        f"Probability of Default: {result['probability_default']:.2%}",
        f"Credit Score: {result['credit_score']}",
        f"Risk Level: {result['risk_level']}",
        f"Decision: {result['decision']}",
        f"Model Version: {result['model_version']}",
        "",
        "Top Risk Factors:",
    ]
    if result["top_factors"]:
        for row in result["top_factors"]:
            lines.append(f"  - {row['feature']}: {row['direction']} (impact={row['impact']:.4f})")
    else:
        lines.append("  - SHAP explanation not available.")
    if result.get("monitoring"):
        lines.append("")
        lines.append("Monitoring Snapshot:")
        lines.append(f"  Alert: {result['monitoring'].get('alert', False)}")
        lines.append(f"  Risk Level: {result['monitoring'].get('risk_level', 'Low')}")
        lines.append(f"  Reasons: {result['monitoring'].get('reasons', [])}")
    return "\n".join(lines)


def create_application_and_score(payload: dict, alternate_payload: dict) -> dict:
    app_id = insert_application(payload)
    upsert_alternate_data(alternate_payload)

    app_record = get_application(app_id)
    result = predict_from_records(app_record, alternate_payload)
    pred_id = insert_prediction(
        {
            "application_id": app_id,
            "customer_id": payload["customer_id"],
            "probability_default": result["probability_default"],
            "credit_score": result["credit_score"],
            "risk_level": result["risk_level"],
            "decision": result["decision"],
            "model_version": result["model_version"],
        }
    )

    monitoring = monitor_customer(payload["customer_id"], application_id=app_id)

    combined = {
        "application_id": app_id,
        "prediction_id": pred_id,
        "name": payload["applicant_name"],
        "customer_id": payload["customer_id"],
        "monitoring": monitoring,
        **result,
    }
    combined["report_text"] = build_report_text(combined)
    return combined


st.set_page_config(page_title="AI Lending Risk Intelligence", layout="wide")
st.title("AI-Powered Lending Risk Intelligence Platform")

init_db()
try:
    ensure_model_artifacts()
except Exception as exc:  # noqa: BLE001
    st.warning("Train model first: `python train_model.py`")
    st.caption(str(exc))
    st.stop()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

tab1, tab2, tab3 = st.tabs(["Loan Application", "Risk Result", "Bank Dashboard"])

with tab1:
    st.subheader("Screen 1 - Loan Application Form")
    c1, c2, c3 = st.columns(3)

    with c1:
        applicant_name = st.text_input("Applicant Name", "Applicant A")
        customer_id = st.text_input("Customer ID (optional)", "")
        age = st.number_input("Age", 18, 80, 30)
        occupation = st.selectbox("Occupation", ["Engineer", "Teacher", "Doctor", "Lawyer", "Businessman", "Other"])
        credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
        has_credit_history = st.selectbox("Has Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        annual_income = st.number_input("Annual Income", 100000, 5000000, 900000, step=10000)
        monthly_salary = st.number_input("Monthly Inhand Salary", 10000, 500000, 70000, step=1000)
        num_of_loan = st.number_input("Existing Loans", 0, 20, 2)
        interest_rate = st.number_input("Interest Rate (%)", 1.0, 36.0, 11.5, step=0.1)
        emi = st.number_input("Total EMI per Month", 0, 300000, 15000, step=500)
        invested_amount = st.number_input("Amount Invested Monthly", 0, 300000, 12000, step=500)

    with c3:
        delayed_payments = st.number_input("No. of Delayed Payments", 0, 60, 3)
        delay_days = st.number_input("Delay from Due Date", 0, 120, 4)
        credit_utilization = st.slider("Credit Utilization Ratio (%)", 1, 100, 35)
        outstanding_debt = st.number_input("Outstanding Debt", 0, 5000000, 120000, step=5000)
        credit_inquiries = st.number_input("No. of Credit Inquiries", 0, 50, 3)
        payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No", "NM"])

    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "Low_spent_Small_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
        ],
    )

    st.markdown("### Alternate Data (Thin-File Inputs)")
    a1, a2, a3, a4 = st.columns(4)
    upi_txn_count_30d = a1.number_input("UPI Txn Count (30d)", 0, 500, 45)
    utility_bill_ontime_ratio = a2.slider("Utility Bill On-time Ratio", 0.0, 1.0, 0.85, 0.01)
    recharge_regularity_score = a3.slider("Recharge Regularity Score", 0.0, 1.0, 0.75, 0.01)
    spending_consistency_score = a4.slider("Spending Consistency Score", 0.0, 1.0, 0.70, 0.01)

    if st.button("Check Eligibility"):
        final_customer_id = customer_id.strip() or f"CUS_{uuid.uuid4().hex[:8]}"
        app_payload = {
            "customer_id": final_customer_id,
            "applicant_name": applicant_name,
            "age": age,
            "occupation": occupation,
            "credit_mix": credit_mix,
            "has_credit_history": has_credit_history,
            "annual_income": annual_income,
            "monthly_salary": monthly_salary,
            "num_of_loan": num_of_loan,
            "interest_rate": interest_rate,
            "emi": emi,
            "delayed_payments": delayed_payments,
            "delay_days": delay_days,
            "credit_utilization": credit_utilization,
            "outstanding_debt": outstanding_debt,
            "invested_amount": invested_amount,
            "credit_inquiries": credit_inquiries,
            "payment_behaviour": payment_behaviour,
            "payment_of_min_amount": payment_of_min_amount,
        }
        alt_payload = {
            "customer_id": final_customer_id,
            "as_of_date": datetime.utcnow().date().isoformat(),
            "upi_txn_count_30d": upi_txn_count_30d,
            "utility_bill_ontime_ratio": utility_bill_ontime_ratio,
            "recharge_regularity_score": recharge_regularity_score,
            "spending_consistency_score": spending_consistency_score,
            "source": "streamlit_form",
        }

        st.session_state.last_result = create_application_and_score(app_payload, alt_payload)
        st.success("Prediction generated and saved to SQL. Open 'Risk Result' tab.")

with tab2:
    st.subheader("Screen 2 - Credit Risk Result")
    if st.session_state.last_result is None:
        st.info("Submit a loan application first.")
    else:
        r = st.session_state.last_result
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score", r["credit_score"])
        c2.metric("Risk Level", r["risk_level"])
        c3.metric("Decision", r["decision"])
        c4.metric("Default Probability", f"{r['probability_default']:.2%}")

        st.markdown("### Explainable AI - Top Factors")
        if r["top_factors"]:
            factors_df = pd.DataFrame(r["top_factors"])
            factors_df["impact"] = factors_df["impact"].round(4)
            st.dataframe(factors_df, width="stretch")
        else:
            st.warning("SHAP explanation unavailable.")
            if r.get("explain_error"):
                st.caption(r["explain_error"])

        st.markdown("### Monitoring Snapshot")
        st.json(r["monitoring"])

        st.download_button(
            label="Download Report",
            data=r["report_text"],
            file_name=f"{safe_filename(r['name'])}_credit_report.txt",
            mime="text/plain",
        )

with tab3:
    st.subheader("Screen 3 - Bank Dashboard")
    st.markdown("### Portfolio Monitoring Sweep")
    sweep_limit = st.number_input("Max Customers to Scan (0 = all)", 0, 100000, 0, step=100)
    if st.button("Run Monitoring Sweep"):
        summary = monitor_all_customers(limit=None if sweep_limit <= 0 else int(sweep_limit))
        st.success(
            f"Checked {summary['customers_checked']} customers | "
            f"Alerts generated: {summary['alerts_generated']}"
        )

    st.markdown("### Record Loan Outcome (Real Label)")
    o1, o2, o3, o4 = st.columns(4)
    out_app_id = o1.number_input("Application ID", 1, 10_000_000, 1)
    out_customer_id = o2.text_input("Outcome Customer ID")
    out_dpd = o3.number_input("Days Past Due", 0, 365, 0)
    out_default_flag = o4.selectbox("Default Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    if st.button("Save Outcome"):
        if not out_customer_id.strip():
            st.error("Outcome Customer ID is required.")
        else:
            outcome_id = insert_outcome(
                {
                    "application_id": int(out_app_id),
                    "customer_id": out_customer_id.strip(),
                    "observed_at": datetime.utcnow().date().isoformat(),
                    "days_past_due": int(out_dpd),
                    "default_flag": int(out_default_flag),
                }
            )
            st.success(f"Outcome saved. ID: {outcome_id}")

    st.markdown("### Ingest Transaction (Time-Series Monitoring)")
    tx1, tx2, tx3, tx4 = st.columns(4)
    tx_customer_id = tx1.text_input("Customer ID", value=(st.session_state.last_result or {}).get("customer_id", ""))
    tx_type = tx2.selectbox("Txn Type", ["UPI", "DEBIT", "SPEND", "PURCHASE", "BILL", "RECHARGE", "EMI_PAYMENT"])
    tx_amount = tx3.number_input("Amount", 0.0, 1000000.0, 1500.0, step=100.0)
    tx_balance = tx4.number_input("Balance After", -1000000.0, 10000000.0, 20000.0, step=100.0)

    tx_is_emi = 1 if tx_type == "EMI_PAYMENT" else 0
    tx_due_date = None
    if tx_is_emi:
        tx_due_date = st.date_input("EMI Due Date").isoformat()

    if st.button("Save Transaction & Run Monitoring"):
        if not tx_customer_id.strip():
            st.error("Customer ID is required.")
        else:
            insert_transaction(
                {
                    "customer_id": tx_customer_id.strip(),
                    "txn_ts": datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
                    "txn_type": tx_type,
                    "amount": tx_amount,
                    "balance_after": tx_balance,
                    "is_emi": tx_is_emi,
                    "emi_due_date": tx_due_date,
                }
            )
            snapshot = monitor_customer(tx_customer_id.strip())
            if snapshot["alert"]:
                st.error(f"ALERT: {snapshot['risk_level']} | Reasons: {snapshot['reasons']}")
            else:
                st.success("Monitoring updated. No alert.")

    st.markdown("### Latest Predictions")
    preds = pd.DataFrame(list_latest_predictions(limit=200))
    st.dataframe(preds, width="stretch")

    st.markdown("### Open Alerts")
    alerts = pd.DataFrame(list_open_alerts(limit=200))
    st.dataframe(alerts, width="stretch")

    st.markdown("### Business Impact Metrics")
    st.dataframe(metrics_frame_for_ui(), width="stretch")
    metrics = compute_business_metrics()
    st.markdown("### Portfolio Risk Distribution")
    risk_dist = metrics.get("portfolio_risk_distribution", {})
    if risk_dist:
        dist_df = pd.DataFrame(
            [{"Risk Level": k, "Share": v * 100.0} for k, v in risk_dist.items()]
        )
        st.bar_chart(dist_df.set_index("Risk Level"))
    else:
        st.info("No prediction distribution available yet.")
