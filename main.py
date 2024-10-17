"""
Script for FastAPI instance and model inference
author: Srinivas Dumpala
Date: Oct. 16th 2024
"""
import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and label binarizer
model_path = "./model/model.pkl"
lb_path = "./model/label_binarizer.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(lb_path, "rb") as lb_file:
    lb = pickle.load(lb_file)

with open("model/encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

# Define the DataInput Pydantic model
class DataInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API Created with FastAPI!"}

# Model inference endpoint
@app.post("/predict")
def predict(data: DataInput):
    # Convert list of DataInput objects to DataFrame
    input_df = pd.DataFrame([{"age": data.age,
                        "workclass": data.workclass,
                        "fnlgt": data.fnlgt,
                        "education": data.education,
                        "education-num": data.education_num,
                        "marital-status": data.marital_status,
                        "occupation": data.occupation,
                        "relationship": data.relationship,
                        "race": data.race,
                        "sex": data.sex,
                        "capital-gain": data.capital_gain,
                        "capital-loss": data.capital_loss,
                        "hours-per-week": data.hours_per_week,
                        "native-country": data.native_country}])


    # Logging the input data for debugging purposes
    logger.info(f" input_data: {input_df.to_dict()}")

    # Process input data
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    # Perform inference
    predictions = inference(model, X)
    # Convert predictions back to original labels
    preds = lb.inverse_transform(predictions)

    logging.info(f" predictions: {preds}")

    # Return predictions
    return {"predictions": preds.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:10000