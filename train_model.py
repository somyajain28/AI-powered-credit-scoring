# train_model.py
import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from database import init_db
from feature_engineering import add_features
from persistence import get_training_frame


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except Exception:  # noqa: BLE001
    HAS_IMBLEARN = False


ORDINAL_FEATURES = ["Credit_Mix", "Payment_of_Min_Amount"]


def load_labeled_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        init_db()
        df = get_training_frame()

    if "default_flag" not in df.columns:
        raise ValueError(
            "Real label `default_flag` is required. "
            "Provide a labeled CSV via --csv-path or insert outcomes into SQLite loan_outcomes table."
        )

    df = df[df["default_flag"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No labeled rows found for training.")
    return df


def run_eda(df: pd.DataFrame, out_dir: str) -> dict:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    missing_pct = (df.isna().mean() * 100.0).round(2).sort_values(ascending=False)
    class_balance = (
        df["default_flag"].value_counts(dropna=False).rename_axis("default_flag").reset_index(name="count")
    )
    class_balance["ratio"] = (class_balance["count"] / max(len(df), 1)).round(4)

    key_vars = [
        "Annual_Income",
        "Total_EMI_per_month",
        "Num_of_Delayed_Payment",
        "Delay_from_due_date",
        "Outstanding_Debt",
    ]
    dist_rows = []
    for c in key_vars:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            dist_rows.append(
                {
                    "feature": c,
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "median": float(series.median()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().any() else None,
                    "p95": float(series.quantile(0.95)) if series.notna().any() else None,
                }
            )
    dist_df = pd.DataFrame(dist_rows)

    corr_df = pd.DataFrame(columns=["feature", "corr_with_default_abs"])
    if numeric_cols and "default_flag" in numeric_cols:
        corr = df[numeric_cols].corr(numeric_only=True)["default_flag"].drop(labels=["default_flag"], errors="ignore")
        corr_df = (
            corr.abs()
            .sort_values(ascending=False)
            .rename("corr_with_default_abs")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    os.makedirs(out_dir, exist_ok=True)
    missing_path = os.path.join(out_dir, "eda_missing_pct.csv")
    class_path = os.path.join(out_dir, "eda_class_balance.csv")
    dist_path = os.path.join(out_dir, "eda_key_distributions.csv")
    corr_path = os.path.join(out_dir, "eda_correlations.csv")
    summary_path = os.path.join(out_dir, "eda_summary.json")

    missing_pct.reset_index().rename(columns={"index": "feature", 0: "missing_pct"}).to_csv(
        missing_path, index=False
    )
    class_balance.to_csv(class_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "class_balance": class_balance.to_dict(orient="records"),
        "top_correlated_features": corr_df.head(10).to_dict(orient="records"),
        "files": {
            "missing_pct": missing_path,
            "class_balance": class_path,
            "key_distributions": dist_path,
            "correlations": corr_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = summary_path
    return summary


def make_onehot_encoder(dense_output: bool = False) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)


def build_preprocessor(feature_cols: list[str], dense_output: bool = False) -> ColumnTransformer:
    ordinal_cols = [c for c in ORDINAL_FEATURES if c in feature_cols]
    all_numeric = []
    for c in feature_cols:
        if c in ordinal_cols:
            continue
        if c in {
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_of_Loan",
            "Interest_Rate",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries",
            "debt_to_income",
            "loan_burden_ratio",
            "payment_consistency_score",
            "spending_volatility",
            "credit_utilization",
            "alt_data_score",
            "alt_payment_behavior_score",
            "balance_drop",
            "behavior_risk_signal",
            "has_credit_history",
            "upi_txn_count_30d",
            "utility_bill_ontime_ratio",
            "recharge_regularity_score",
            "spending_consistency_score",
        }:
            all_numeric.append(c)
    nominal_cols = [c for c in feature_cols if c not in set(all_numeric + ordinal_cols)]

    transformers = [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), all_numeric),
    ]
    if ordinal_cols:
        # Keep ordinal meaning in numeric form instead of one-hot for ranked categories.
        ord_categories = []
        for c in ordinal_cols:
            if c == "Credit_Mix":
                ord_categories.append(["Bad", "Standard", "Good"])
            elif c == "Payment_of_Min_Amount":
                ord_categories.append(["No", "NM", "Yes"])
            else:
                ord_categories.append("auto")
        transformers.append(
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=ord_categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ordinal_cols,
            )
        )
    if nominal_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", make_onehot_encoder(dense_output=dense_output)),
                    ]
                ),
                nominal_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def make_model_pipeline(prep: ColumnTransformer, clf, imbalance_method: str):
    prep = clone(prep)
    if imbalance_method == "smote":
        if not HAS_IMBLEARN:
            raise ImportError(
                "SMOTE requested but `imbalanced-learn` is not installed. "
                "Install with `pip install imbalanced-learn` or use --imbalance-method class_weight."
            )
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=42)), ("clf", clf)])
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    pd_hat = model.predict_proba(x_test)[:, 1]
    y_hat = (pd_hat >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_hat)),
        "precision": float(precision_score(y_test, y_hat, zero_division=0)),
        "recall": float(recall_score(y_test, y_hat, zero_division=0)),
        "f1": float(f1_score(y_test, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, pd_hat)),
    }
    fpr, tpr, _ = roc_curve(y_test, pd_hat)
    return metrics, fpr, tpr


def fit_model(model_name: str, model, x_train: pd.DataFrame, y_train: pd.Series, imbalance_method: str) -> None:
    if model_name == "deep_mlp" and imbalance_method == "class_weight":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        try:
            model.fit(x_train, y_train, clf__sample_weight=sample_weight)
            return
        except TypeError:
            print(
                "Warning: sample_weight not supported for this sklearn MLP version. "
                "Training deep_mlp without class weighting."
            )
    model.fit(x_train, y_train)


def select_best_model(results: dict[str, dict]) -> str:
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["roc_auc"], kv[1]["f1"], kv[1]["recall"], kv[1]["precision"]),
        reverse=True,
    )
    return ranked[0][0]


def main():
    parser = argparse.ArgumentParser(description="Train risk model on real default labels.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional labeled CSV path. Must include default_flag column.",
    )
    parser.add_argument(
        "--imbalance-method",
        type=str,
        default="class_weight",
        choices=["class_weight", "smote", "none"],
        help="How to handle class imbalance. SMOTE is applied only on training data.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA report generation.",
    )
    args = parser.parse_args()

    df = load_labeled_data(args.csv_path)
    if not args.skip_eda:
        eda = run_eda(df, ART_DIR)
        print("EDA saved to:", eda["summary_json"])
    df = add_features(df)

    y = df["default_flag"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target has only one class. Need both default_flag=0 and default_flag=1.")

    feature_cols = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "debt_to_income",
        "loan_burden_ratio",
        "payment_consistency_score",
        "spending_volatility",
        "credit_utilization",
        "alt_data_score",
        "alt_payment_behavior_score",
        "balance_drop",
        "behavior_risk_signal",
        "has_credit_history",
        "upi_txn_count_30d",
        "utility_bill_ontime_ratio",
        "recharge_regularity_score",
        "spending_consistency_score",
        "Occupation",
        "Credit_Mix",
        "Payment_Behaviour",
        "Payment_of_Min_Amount",
    ]
    X = df[feature_cols].copy()
    preprocessor = build_preprocessor(
        feature_cols,
        dense_output=(args.imbalance_method == "smote"),
    )

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if args.imbalance_method == "class_weight" else 1.0
    lr_weight = "balanced" if args.imbalance_method == "class_weight" else None

    lr = make_model_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000, class_weight=lr_weight),
        args.imbalance_method,
    )
    rf = make_model_pipeline(
        preprocessor,
        RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            class_weight=("balanced_subsample" if args.imbalance_method == "class_weight" else None),
            n_jobs=-1,
        ),
        args.imbalance_method,
    )
    xgb = make_model_pipeline(
        preprocessor,
        XGBClassifier(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        ),
        args.imbalance_method,
    )
    mlp_preprocessor = build_preprocessor(
        feature_cols,
        dense_output=True,
    )
    deep_mlp = make_model_pipeline(
        mlp_preprocessor,
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=True,
            random_state=42,
        ),
        args.imbalance_method,
    )

    models = {
        "logistic": lr,
        "random_forest": rf,
        "xgb": xgb,
        "deep_mlp": deep_mlp,
    }
    metrics_by_model: dict[str, dict] = {}
    roc_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, model in models.items():
        fit_model(name, model, x_train, y_train, args.imbalance_method)
        m, fpr, tpr = evaluate_model(model, x_test, y_test)
        metrics_by_model[name] = m
        roc_by_model[name] = (fpr, tpr)

    best_model_name = select_best_model(metrics_by_model)
    best_model = models[best_model_name]

    print("Rows used :", len(df))
    print("Imbalance method:", args.imbalance_method)
    print("Model comparison:")
    for name, m in metrics_by_model.items():
        print(
            f"  {name:14s} "
            f"ACC={m['accuracy']:.4f} "
            f"PREC={m['precision']:.4f} "
            f"REC={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"AUC={m['roc_auc']:.4f}"
        )
    print("Selected best model:", best_model_name)

    model_version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    joblib.dump(models["logistic"], os.path.join(ART_DIR, "logistic.pkl"))
    joblib.dump(models["random_forest"], os.path.join(ART_DIR, "random_forest.pkl"))
    joblib.dump(models["xgb"], os.path.join(ART_DIR, "xgb.pkl"))
    joblib.dump(models["deep_mlp"], os.path.join(ART_DIR, "deep_mlp.pkl"))
    joblib.dump(best_model, os.path.join(ART_DIR, "best_model.pkl"))
    joblib.dump(best_model_name, os.path.join(ART_DIR, "best_model_name.pkl"))
    joblib.dump(feature_cols, os.path.join(ART_DIR, "feature_cols.pkl"))
    joblib.dump(model_version, os.path.join(ART_DIR, "model_version.pkl"))

    comparison_df = (
        pd.DataFrame(metrics_by_model)
        .T.reset_index()
        .rename(columns={"index": "model"})
        .sort_values(["roc_auc", "f1"], ascending=False)
    )
    comparison_path = os.path.join(ART_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print("Saved model comparison to:", comparison_path)

    for name, (fpr, tpr) in roc_by_model.items():
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(os.path.join(ART_DIR, f"roc_curve_{name}.csv"), index=False)

    print("Saved artifacts to:", ART_DIR)
    print("Model version:", model_version)


if __name__ == "__main__":
    main()
