from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "loan_approval_dataset.csv"
MODEL_PATH = BASE_DIR / "random_forest_loan_model.pkl"
EVALUATION_PATH = BASE_DIR / "model_evaluation_results.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance_results.csv"

FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_FEATURES = ["education", "self_employed"]

FEATURE_LABELS = {
    "no_of_dependents": "Jumlah Tanggungan",
    "education": "Pendidikan",
    "self_employed": "Self Employed",
    "income_annum": "Pendapatan Tahunan",
    "loan_amount": "Jumlah Pinjaman",
    "loan_term": "Tenor Pinjaman",
    "cibil_score": "CIBIL Score",
    "residential_assets_value": "Aset Residensial",
    "commercial_assets_value": "Aset Komersial",
    "luxury_assets_value": "Aset Mewah",
    "bank_asset_value": "Aset Bank",
    "education_Graduate": "Pendidikan: Graduate",
    "education_Not Graduate": "Pendidikan: Not Graduate",
    "self_employed_No": "Self Employed: No",
    "self_employed_Yes": "Self Employed: Yes",
}

app = Flask(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan nama kolom dan isi teks supaya konsisten dengan proses training."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")
    return clean_dataframe(pd.read_csv(DATASET_PATH))


def make_one_hot_encoder():
    """Menjaga kompatibilitas untuk beberapa versi scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def save_feature_importance(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    onehot_encoder = preprocessor.named_transformers_["cat"]
    try:
        categorical_names = onehot_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    except AttributeError:
        categorical_names = onehot_encoder.get_feature_names(CATEGORICAL_FEATURES)

    feature_names = np.concatenate([NUMERIC_FEATURES, categorical_names])

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": classifier.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def train_random_forest_model() -> Pipeline:
    """Melatih ulang model jika file pkl tidak kompatibel dengan environment lokal."""
    df = load_dataset()

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    X = df[FEATURE_COLUMNS]
    y = df["loan_status"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    rf_metrics = {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }

    if EVALUATION_PATH.exists():
        evaluation_df = pd.read_csv(EVALUATION_PATH)
        evaluation_df.columns = evaluation_df.columns.str.strip()
        evaluation_df = evaluation_df[evaluation_df["Model"].str.lower() != "random forest"]
        evaluation_df = pd.concat([pd.DataFrame([rf_metrics]), evaluation_df], ignore_index=True)
    else:
        evaluation_df = pd.DataFrame([rf_metrics])

    evaluation_df.to_csv(EVALUATION_PATH, index=False)
    save_feature_importance(model)

    return model


def validate_loaded_model(model: Pipeline) -> None:
    sample = pd.DataFrame(
        [
            {
                "no_of_dependents": 1,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 8000000,
                "loan_amount": 20000000,
                "loan_term": 8,
                "cibil_score": 780,
                "residential_assets_value": 9000000,
                "commercial_assets_value": 3000000,
                "luxury_assets_value": 12000000,
                "bank_asset_value": 5000000,
            }
        ]
    )

    model.predict(sample)
    if hasattr(model, "predict_proba"):
        model.predict_proba(sample)


def load_or_train_model():
    try:
        loaded_model = joblib.load(MODEL_PATH)
        validate_loaded_model(loaded_model)
        return loaded_model, "Model Random Forest berhasil dimuat dari random_forest_loan_model.pkl."
    except Exception as error:
        trained_model = train_random_forest_model()
        return (
            trained_model,
            "Model pkl tidak kompatibel dengan environment lokal, sehingga sistem melatih ulang model dari dataset. "
            f"Detail: {error}",
        )


model, model_status = load_or_train_model()


def label_feature(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def load_evaluation_results():
    if not EVALUATION_PATH.exists():
        return []

    df = pd.read_csv(EVALUATION_PATH)
    df.columns = df.columns.str.strip()

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "model": str(row.get("Model", "-")),
                "accuracy": float(row.get("Accuracy", 0)),
                "precision": float(row.get("Precision", 0)),
                "recall": float(row.get("Recall", 0)),
                "f1": float(row.get("F1-Score", 0)),
            }
        )

    return records


def load_feature_importance(limit=None):
    if not FEATURE_IMPORTANCE_PATH.exists():
        return []

    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    df.columns = df.columns.str.strip()
    df = df.sort_values(by="Importance", ascending=False)
    if limit:
        df = df.head(limit)

    max_value = float(df["Importance"].max()) if not df.empty else 1

    records = []
    for _, row in df.iterrows():
        importance = float(row["Importance"])
        feature = str(row["Feature"])
        records.append(
            {
                "feature": feature,
                "label": label_feature(feature),
                "importance": importance,
                "percentage": round((importance / max_value) * 100, 2) if max_value else 0,
            }
        )

    return records


def get_default_inputs(df: pd.DataFrame):
    return {
        "no_of_dependents": int(df["no_of_dependents"].median()),
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": int(df["income_annum"].median()),
        "loan_amount": int(df["loan_amount"].median()),
        "loan_term": int(df["loan_term"].median()),
        "cibil_score": int(df["cibil_score"].median()),
        "residential_assets_value": int(df["residential_assets_value"].median()),
        "commercial_assets_value": int(df["commercial_assets_value"].median()),
        "luxury_assets_value": int(df["luxury_assets_value"].median()),
        "bank_asset_value": int(df["bank_asset_value"].median()),
    }


def calculate_outliers(df: pd.DataFrame):
    outlier_counts = {}
    boxplot_data = {}

    for col in NUMERIC_FEATURES:
        values = df[col].dropna().astype(float)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        outlier_counts[col] = int(mask.sum())
        boxplot_data[col] = {
            "label": label_feature(col),
            "q1": q1,
            "median": float(values.median()),
            "q3": q3,
            "whisker_low": float(max(values.min(), lower)),
            "whisker_high": float(min(values.max(), upper)),
        }

    return outlier_counts, boxplot_data


def get_model_report(df: pd.DataFrame):
    work_df = df.copy()
    y = work_df["loan_status"].map({"Approved": 1, "Rejected": 0})
    X = work_df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Rejected", "Approved"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "confusion_matrix": cm,
        "classification_report": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in ["Rejected", "Approved"]
        },
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
    }


def compute_analytics():
    df = load_dataset()

    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    target_counts = df["loan_status"].value_counts().to_dict()
    approved_count = int(target_counts.get("Approved", 0))
    rejected_count = int(target_counts.get("Rejected", 0))
    approval_rate = round((approved_count / total_rows) * 100, 2) if total_rows else 0

    evaluation = load_evaluation_results()
    random_forest_eval = next(
        (item for item in evaluation if item["model"].lower() == "random forest"),
        evaluation[0] if evaluation else None,
    )
    accuracy = round(float(random_forest_eval["accuracy"]) * 100, 2) if random_forest_eval else 0

    df_corr = df.copy()
    df_corr["target_encoded"] = (df_corr["loan_status"] == "Approved").astype(int)
    corr = (
        df_corr[NUMERIC_FEATURES + ["target_encoded"]]
        .corr(numeric_only=True)["target_encoded"]
        .drop("target_encoded")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    correlation = [
        {"feature": col, "label": label_feature(col), "value": round(float(val), 4)}
        for col, val in corr.items()
    ]

    outlier_counts, boxplot_data = calculate_outliers(df)

    distribution_data = {}
    for feature in ["cibil_score", "loan_amount", "income_annum", "loan_term"]:
        distribution_data[feature] = {
            "label": label_feature(feature),
            "Approved": df[df["loan_status"] == "Approved"][feature].astype(float).tolist(),
            "Rejected": df[df["loan_status"] == "Rejected"][feature].astype(float).tolist(),
        }

    scatter_approved = df[df["loan_status"] == "Approved"].sample(
        n=min(500, approved_count),
        random_state=42,
    )
    scatter_rejected = df[df["loan_status"] == "Rejected"].sample(
        n=min(500, rejected_count),
        random_state=42,
    )

    scatter_data = {
        "Approved": [
            {"x": float(row["cibil_score"]), "y": float(row["loan_amount"])}
            for _, row in scatter_approved.iterrows()
        ],
        "Rejected": [
            {"x": float(row["cibil_score"]), "y": float(row["loan_amount"])}
            for _, row in scatter_rejected.iterrows()
        ],
    }

    desc = df[NUMERIC_FEATURES].describe().round(2).to_dict()
    descriptive_stats = [
        {
            "feature": col,
            "label": label_feature(col),
            "mean": float(desc[col]["mean"]),
            "median": float(df[col].median()),
            "min": float(desc[col]["min"]),
            "max": float(desc[col]["max"]),
        }
        for col in NUMERIC_FEATURES
    ]

    model_report = get_model_report(df)

    return {
        "model_status": model_status,
        "dataset_info": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "approval_rate": approval_rate,
            "rejection_rate": round(100 - approval_rate, 2),
        },
        "target_counts": target_counts,
        "accuracy": accuracy,
        "defaults": get_default_inputs(df),
        "evaluation": evaluation,
        "feature_importance": load_feature_importance(),
        "correlation": correlation,
        "outlier_counts": [
            {"feature": col, "label": label_feature(col), "count": count}
            for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
        ],
        "boxplot_data": boxplot_data,
        "distribution_data": distribution_data,
        "scatter_data": scatter_data,
        "descriptive_stats": descriptive_stats,
        **model_report,
    }


analytics_cache = compute_analytics()


def parse_prediction_payload(payload):
    try:
        parsed = {
            "no_of_dependents": int(payload.get("no_of_dependents")),
            "education": str(payload.get("education", "")).strip(),
            "self_employed": str(payload.get("self_employed", "")).strip(),
            "income_annum": int(payload.get("income_annum")),
            "loan_amount": int(payload.get("loan_amount")),
            "loan_term": int(payload.get("loan_term")),
            "cibil_score": int(payload.get("cibil_score")),
            "residential_assets_value": int(payload.get("residential_assets_value")),
            "commercial_assets_value": int(payload.get("commercial_assets_value")),
            "luxury_assets_value": int(payload.get("luxury_assets_value")),
            "bank_asset_value": int(payload.get("bank_asset_value")),
        }
    except (TypeError, ValueError):
        raise ValueError("Input numerik wajib diisi dengan angka yang valid.")

    if parsed["education"] not in ["Graduate", "Not Graduate"]:
        raise ValueError("Education hanya boleh Graduate atau Not Graduate.")

    if parsed["self_employed"] not in ["Yes", "No"]:
        raise ValueError("Self Employed hanya boleh Yes atau No.")

    if not 0 <= parsed["no_of_dependents"] <= 10:
        raise ValueError("Jumlah tanggungan sebaiknya berada pada rentang 0 sampai 10.")

    if not 300 <= parsed["cibil_score"] <= 900:
        raise ValueError("CIBIL score sebaiknya berada pada rentang 300 sampai 900.")

    if parsed["loan_term"] <= 0:
        raise ValueError("Loan term harus lebih dari 0.")

    for key in [
        "income_annum",
        "loan_amount",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]:
        if parsed[key] < 0:
            raise ValueError("Nilai aset, pendapatan, dan pinjaman tidak boleh negatif.")

    return parsed


def make_prediction_note(input_data, prediction):
    notes = []
    if input_data["cibil_score"] < 650:
        notes.append("CIBIL score relatif rendah, sehingga risiko penolakan dapat meningkat.")
    if input_data["loan_amount"] > input_data["income_annum"] * 4:
        notes.append("Jumlah pinjaman cukup besar dibandingkan pendapatan tahunan.")
    if input_data["loan_term"] > 15:
        notes.append("Tenor pinjaman panjang perlu diperhatikan karena dapat memengaruhi profil risiko.")
    if not notes:
        notes.append("Profil input berada pada pola yang relatif wajar berdasarkan fitur yang digunakan model.")

    if prediction == "Approved":
        prefix = "Model memprediksi pengajuan berpeluang disetujui."
    else:
        prefix = "Model memprediksi pengajuan berisiko ditolak."

    return prefix + " " + " ".join(notes)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analytics")
def api_analytics():
    return jsonify(analytics_cache)


@app.route("/api/dataset-sample")
def api_dataset_sample():
    df = load_dataset().head(15)
    return jsonify(
        {
            "columns": df.columns.tolist(),
            "rows": df.replace({np.nan: None}).to_dict(orient="records"),
        }
    )


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        input_data = parse_prediction_payload(payload)

        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]
            probability_rejected = float(probability[0])
            probability_approved = float(probability[1])
        else:
            probability_rejected = 1.0 if prediction == 0 else 0.0
            probability_approved = 1.0 if prediction == 1 else 0.0

        status = "Approved" if prediction == 1 else "Rejected"

        return jsonify(
            {
                "status": "success",
                "prediction": status,
                "probability_rejected": round(probability_rejected * 100, 2),
                "probability_approved": round(probability_approved * 100, 2),
                "note": make_prediction_note(input_data, status),
                "input": input_data,
            }
        )
    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 400


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_status": model_status,
            "features": FEATURE_COLUMNS,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
