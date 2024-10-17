"""
Unit test of model.py module with pytest
author: Srinivas Dumpala
Date: Oct. 16th 2024
"""

import pytest
from fastapi.testclient import TestClient
from main import app  # Ensure this imports your FastAPI app
import json

client = TestClient(app)

def test_get_root():
    response = client.get("https://deploying-a-scalable-ml-pipeline-in.onrender.com/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Model API Created with FastAPI!"}

def test_post_predict_1():
    sample_data = {
            "age": 49,
            "workclass": "Private",
            "fnlgt": 160187,
            "education": "9th",
            "education_num": 5,
            "marital_status": "Married-spouse-absent",
            "occupation": "Other-service",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 16,
            "native_country": "Jamaica"
        }
        
    response = client.post("https://deploying-a-scalable-ml-pipeline-in.onrender.com/predict", json=sample_data)
    assert response.status_code == 200
    print(response.json()["predictions"])
    assert response.json()["predictions"] == [" <=50K"]

def test_post_predict_2():
    sample_data = {
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["predictions"] == [" >50K"]