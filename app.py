from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import pandas as pd
import tensorflow as tf

app = Flask(__name__, template_folder="templates", static_folder="static")

PREPC_PATH = os.path.join("model", "titanic_preprocessor.joblib")
MODEL_PATH = os.path.join("model", "titanic_model.h5")

# ---Load preprocessor + model
if not os.path.exists(PREPC_PATH) or not os.path.exists(MODEL_PATH):
    print("⚠️ Preprocessor or model not found. Please run train_titanic.py first.")

preprocessor = joblib.load(PREPC_PATH) if os.path.exists(PREPC_PATH) else None
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if preprocessor is None or model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500

    # Accept JSON or form
    data = request.get_json(silent=True) or request.form

    try:
        row = {
            "Pclass": int(data.get("Pclass", 3)),
            "Sex": str(data.get("Sex", "female")).lower(),
            "Age": float(data.get("Age")) if data.get("Age") else np.nan,
            "SibSp": int(data.get("SibSp", 0)),
            "Parch": int(data.get("Parch", 0)),
            "Fare": float(data.get("Fare", 0.0)),
            "Embarked": str(data.get("Embarked", "S")).upper()
        }
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid input format"}), 400

    df = pd.DataFrame([row])

    # Preprocess numerical and categorical features
    num_imputer = preprocessor["num_imputer"]
    scaler = preprocessor["scaler"]
    cat_imputer = preprocessor["cat_imputer"]
    encoder = preprocessor["encoder"]
    numfts = preprocessor["num_features"]
    catfts = preprocessor["cat_features"]

    X_num = num_imputer.transform(df[numfts])
    X_num_scaled = scaler.transform(X_num)

    X_cat = cat_imputer.transform(df[catfts])
    X_cat_encoded = encoder.transform(X_cat)

    X_cleaned = np.hstack([X_num_scaled, X_cat_encoded])

    # Predict probability
    proba = float(model.predict(X_cleaned, verbose=0)[0][0])

    return jsonify({
        "ok": True,
        "prediction": "Survived" if proba >= 0.5 else "Did Not Survive",
        "probability": round(float(proba), 4),
        "inputs": row
    })

if __name__ == "__main__":
    app.run(debug=True)