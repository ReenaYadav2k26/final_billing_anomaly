from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# LOAD MODEL ARTIFACTS
# -----------------------------
try:
    model = pickle.load(open("model/isolation_forest_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    features = pickle.load(open("model/features.pkl", "rb"))
    
    # Optional threshold
    if os.path.exists("model/threshold.pkl"):
        threshold = pickle.load(open("model/threshold.pkl", "rb"))
    else:
        threshold = -0.05  # fallback

except Exception as e:
    raise Exception(f"Error loading model files: {e}")

# -----------------------------
# API KEY (use env in prod)
# -----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key-123")

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_input(df):
    try:
        # Convert dates (optional)
        date_cols = ['due_date','paid_date','bill_from_date','bill_thru_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Add missing features
        for col in features:
            if col not in df.columns:
                df[col] = 0

        # Keep only training features
        df = df[features]

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN
        df = df.fillna(0)

        return df

    except Exception as e:
        raise Exception(f"Preprocessing error: {e}")

# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # API KEY check
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()

        if not data or "invoice_features" not in data:
            return jsonify({"error": "Missing 'invoice_features'"}), 400

        df = pd.DataFrame([data["invoice_features"]])

        # Preprocess
        df_processed = preprocess_input(df)

        # Scale
        X_scaled = scaler.transform(df_processed)

        # Score
        score = model.decision_function(X_scaled)[0]

        return jsonify({
            "anomaly_score": float(score),
            "is_anomaly": bool(score < threshold)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Invoice Anomaly API Running"})

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
