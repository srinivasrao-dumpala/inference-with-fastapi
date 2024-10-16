"""
Unit test of main.py API module with pytest
author: Srinivas Dumpala
Date: Oct. 16th 2024
"""

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    return pd.read_csv("data/census.csv")

@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

@pytest.fixture
def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test

def test_process_data_training(split_data, cat_features):
    train, _ = split_data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert encoder is not None
    assert lb is not None

def test_process_data_inference(split_data, cat_features):
    train, test = split_data
    _, _, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # assert if the number of samples is equal to the number of labels
    assert X_test.shape[0] == y_test.shape[0]

def test_train_model(split_data, cat_features):
    train, _ = split_data
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert model is not None

def test_inference(split_data, cat_features):
    train, test = split_data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    # assert if the number of predictions is equal to the number of test samples
    assert preds.shape[0] == y_test.shape[0]

def test_compute_model_metrics(split_data, cat_features):
    train, test = split_data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    # assert if the metrics are between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_model_persistence(split_data, cat_features):
    train, test = split_data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    model_filename = "model/model.pkl"
    lb_filename = "model/label_binarizer.pkl"
    
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(lb_filename, "wb") as lb_file:
        pickle.dump(lb, lb_file)
    
    with open(model_filename, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open(lb_filename, "rb") as lb_file:
        loaded_lb = pickle.load(lb_file)
    
    # assert if they are the same type
    assert type(model) == type(loaded_model)
    assert type(lb) == type(loaded_lb)