from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# 🔹 LOAD MODEL ARTIFACTS
# -----------------------------
try:
    model = pickle.load(open("model/isolation_forest_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    features = pickle.load(open("model/features.pkl", "rb"))

    # Load threshold if exists
    if os.path.exists("model/threshold.pkl"):
        threshold = pickle.load(open("model/threshold.pkl", "rb"))
    else:
        threshold = -0.05  # fallback

except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")

# -----------------------------
# 🔐 API KEY (env-based)
# -----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key-123")

# -----------------------------
# 🔹 PREPROCESS FUNCTION
# -----------------------------
def preprocess_input(df):
    try:
        # Date conversion (optional)
        date_cols = ['due_date', 'paid_date', 'bill_from_date', 'bill_thru_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Add missing features
        for col in features:
            if col not in df.columns:
                df[col] = 0

        # Keep only trained features
        df = df[features]

        # Clean values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        return df

    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

# -----------------------------
# 🔹 PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔐 API KEY VALIDATION
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()

        # Validate input
        if not data or "invoice_features" not in data:
            return jsonify({"error": "Missing 'invoice_features'"}), 400

        if not isinstance(data["invoice_features"], dict):
            return jsonify({"error": "Invalid format for invoice_features"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data["invoice_features"]])

        # Preprocess
        df_processed = preprocess_input(df)

        # Scale
        X_scaled = scaler.transform(df_processed)

        # Score
        score = model.decision_function(X_scaled)[0]

        # Result
        result = {
            "anomaly_score": float(score),
            "is_anomaly": bool(score < threshold)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------
# 🔹 HEALTH CHECK
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Invoice Anomaly Detection API"
    })

# -----------------------------
# 🔹 MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Render compatibility
    app.run(host="0.0.0.0", port=port, debug=False)
