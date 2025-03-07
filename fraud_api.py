import os
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
import joblib  # Load the trained model

# Initialize Flask app
app = Flask(__name__)

# Load trained fraud detection model
try:
    model = joblib.load("final_fraud_detection_model_fixed.pkl")  # Ensure correct model name
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# Ensure transactions table exists in SQLite
def verify_table():
    conn = sqlite3.connect("fraud_detection.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type INTEGER,
            amount REAL,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            balance_difference REAL,
            receiver_balance_increase REAL,
            is_sender_empty INTEGER,
            fraud_prediction INTEGER,
            fraud_probability REAL
        )
    ''')
    conn.commit()
    conn.close()

# Create table when the API starts
verify_table()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded!"}), 500

        data = request.get_json()
        required_features = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                     "oldbalanceDest", "newbalanceDest"]

        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing field: {feature}"}), 400

        df = pd.DataFrame([data])
        df = df[required_features]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Insert transaction data into SQLite database
        conn = sqlite3.connect("fraud_detection.db")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (type, amount, oldbalanceOrg, newbalanceOrig, 
                                      oldbalanceDest, newbalanceDest, balance_difference, 
                                      receiver_balance_increase, is_sender_empty, 
                                      fraud_prediction, fraud_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data["type"], data["amount"], data["oldbalanceOrg"], data["newbalanceOrig"], 
              data["oldbalanceDest"], data["newbalanceDest"], data["balance_difference"], 
              data["receiver_balance_increase"], data["is_sender_empty"], 
              int(prediction), float(probability)))

        conn.commit()
        conn.close()

        return jsonify({
            "fraud_prediction": int(prediction),
            "fraud_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app (Render-compatible port handling)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)
