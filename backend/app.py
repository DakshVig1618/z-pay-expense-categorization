# ==================================================
# Z-PAY AI TRANSACTION ANALYSIS API
# ==================================================
# This module exposes machine learning models through
# a REST API using FastAPI.
#
# The API provides the following capabilities:
#
# 1. Predict expense category from transaction text
# 2. Detect fraud risk in transactions
# 3. Perform full transaction analysis
#
# The API loads pre-trained models and encoders
# generated during the ML training pipeline.
# ==================================================


# ==================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==================================================

import os
import pickle
import datetime
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ==================================================
# 2. INITIALIZE FASTAPI APPLICATION
# ==================================================
# FastAPI creates the web server that will expose
# our machine learning models as REST endpoints.

app = FastAPI()


# ==================================================
# 3. HEALTH CHECK ENDPOINT
# ==================================================
# This endpoint confirms that the API server is running.
# It is useful for testing deployment or monitoring.

@app.get("/")
def home():
    return {"message": "Welcome to Z_PAY"}


# ==================================================
# 4. DEFINE PROJECT BASE DIRECTORY
# ==================================================
# The API must locate trained models stored in the
# project folder structure. We determine the root
# directory of the project dynamically.

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# ==================================================
# 5. DEFINE PATHS TO TRAINED MODELS
# ==================================================
# These models were generated during the ML training
# pipeline and saved to the models/trained directory.

expense_model_path = os.path.join(
    BASE_DIR, "models", "trained", "expense_model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR, "models", "trained", "vectorizer.pkl"
)

fraud_model_path = os.path.join(
    BASE_DIR, "models", "trained", "fraud_model.pkl"
)


# ==================================================
# 6. DEFINE PATHS TO ENCODERS
# ==================================================
# During training, categorical variables were converted
# into numerical representations using LabelEncoder.
# These encoders must be reused here so that the API
# applies the same transformations used during training.

category_encoder_path = os.path.join(
    BASE_DIR, "models", "trained", "category_encoder.pkl"
)

location_encoder_path = os.path.join(
    BASE_DIR, "models", "trained", "location_encoder.pkl"
)

device_encoder_path = os.path.join(
    BASE_DIR, "models", "trained", "device_encoder.pkl"
)

# ==================================================
# 7. DEFINED PATH TO LOG FILE
# ==================================================

LOG_FILE = os.path.join(
    BASE_DIR, "data", "logs", "transaction_logs.csv"
)

# ==================================================
# 8. LOAD TRAINED MACHINE LEARNING MODELS
# ==================================================
# Models are loaded using pickle so they can be used
# for real-time predictions through the API.

with open(expense_model_path, "rb") as f:
    expense_model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(fraud_model_path, "rb") as f:
    fraud_model = pickle.load(f)

# ==================================================
# 9. SAVE TRANSACTIONS TO LOG FILE
# ==================================================
# Keeps the record of all the transactions

def log_transaction(data, category, risk_score, is_fraud):

    log_entry = pd.DataFrame([{
        "description": data.description,
        "amount": data.amount,
        "category": category,
        "risk_score": risk_score,
        "is_fraud": is_fraud,
        "location": data.location,
        "device_id": data.device_id,
        "timestamp": datetime.datetime.now()
    }])

    try:
        log_entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    except FileNotFoundError:
        log_entry.to_csv(LOG_FILE, index=False)

# ==================================================
# 10. LOAD LABEL ENCODERS
# ==================================================
# Encoders convert categorical variables into numeric
# values so that the fraud model can process them.

with open(category_encoder_path, "rb") as f:
    category_encoder = pickle.load(f)

with open(location_encoder_path, "rb") as f:
    location_encoder = pickle.load(f)

with open(device_encoder_path, "rb") as f:
    device_encoder = pickle.load(f)

print("All models and encoders loaded successfully")


# ==================================================
# 11. DEFINE REQUEST SCHEMAS
# ==================================================
# Pydantic models define the structure of incoming
# JSON data sent to the API endpoints.

class CategoryInput(BaseModel):
    description: str


class FraudInput(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    category: str
    location: str
    device_id: str


class TransactionInput(BaseModel):
    description: str
    amount: float
    hour: int
    day_of_week: int
    location: str
    device_id: str


# ==================================================
# 12. CATEGORY PREDICTION ENDPOINT
# ==================================================
# This endpoint predicts the spending category of a
# transaction using the NLP expense classification model.

@app.post("/predict_category")
def predict_category(data: CategoryInput):

    try:

        # Convert transaction text into TF-IDF features
        text_features = vectorizer.transform([data.description])

        # Predict category using trained classifier
        prediction = expense_model.predict(text_features)

        category = prediction[0]

        return {
            "description": data.description,
            "predicted_category": category
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected server error")


# ==================================================
# 13. FRAUD DETECTION ENDPOINT
# ==================================================
# This endpoint evaluates a transaction and returns
# the probability that it is fraudulent.

@app.post("/fraud_check")
def fraud_check(data: FraudInput):

    try:

        # Encode categorical variables
        category_encoded = category_encoder.transform([data.category])[0]
        location_encoded = location_encoder.transform([data.location])[0]
        device_encoded = device_encoder.transform([data.device_id])[0]

        # Build feature dataframe expected by fraud model
        df = pd.DataFrame([{
            "amount": data.amount,
            "hour": data.hour,
            "day_of_week": data.day_of_week,
            "category_encoded": category_encoded,
            "location_encoded": location_encoded,
            "device_encoded": device_encoded
        }])

        # Generate fraud probability score
        risk_score = fraud_model.predict_proba(df)[0][1]

        # Flag transaction if probability exceeds threshold
        is_fraud = risk_score > 0.7

        return {
            "risk_score": float(risk_score),
            "is_fraud": bool(is_fraud)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected server error")


# ==================================================
# 14. FULL TRANSACTION ANALYSIS ENDPOINT
# ==================================================
# This endpoint combines both models:
#
# Step 1 → Predict transaction category
# Step 2 → Run fraud detection model

@app.post("/analyze_transaction")
def analyze_transaction(data: TransactionInput):

    try:

        # Predict category from transaction description
        text_features = vectorizer.transform([data.description])
        category_prediction = expense_model.predict(text_features)[0]

        # Encode categorical signals
        category_encoded = category_encoder.transform([category_prediction])[0]
        location_encoded = location_encoder.transform([data.location])[0]
        device_encoded = device_encoder.transform([data.device_id])[0]

        # Prepare fraud model input
        df = pd.DataFrame([{
            "amount": data.amount,
            "hour": data.hour,
            "day_of_week": data.day_of_week,
            "category_encoded": category_encoded,
            "location_encoded": location_encoded,
            "device_encoded": device_encoded
        }])

        # Predict fraud probability
        risk_score = fraud_model.predict_proba(df)[0][1]
        is_fraud = risk_score > 0.7

        # Save transaction to logs
        log_transaction(data, category_prediction, risk_score, is_fraud)

        return {
            "description": data.description,
            "predicted_category": category_prediction,
            "risk_score": float(risk_score),
            "is_fraud": bool(is_fraud)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected server error")