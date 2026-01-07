from flask import Flask, request, jsonify
import pandas as pd
import joblib
import uuid
import datetime
import os

MODEL_PATH = "model/trained_model.pkl"
ENCODER_PATH = "model/encoder.pkl"
LIVE_DATA_PATH = "data/live/live_predictions.csv"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

@app.route("/")
def home():
    return {"status": "SentinelML API running"}

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json

    df = pd.DataFrame([payload])
    df_encoded = encoder.transform(df)

    prediction = model.predict(df_encoded)[0]
    proba = model.predict_proba(df_encoded)[0][1]

    record = payload.copy()
    record["prediction"] = int(prediction)
    record["probability"] = float(proba)
    record["timestamp"] = datetime.datetime.utcnow()
    record["id"] = str(uuid.uuid4())

    os.makedirs(os.path.dirname(LIVE_DATA_PATH), exist_ok=True)

    pd.DataFrame([record]).to_csv(
        LIVE_DATA_PATH,
        mode="a",
        header=not os.path.exists(LIVE_DATA_PATH),
        index=False
    )

    return jsonify({
        "prediction": int(prediction),
        "probability": float(proba)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
