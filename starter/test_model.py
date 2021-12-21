from .ml.model import *
from .ml.data import *

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pytest
import os

@pytest.fixture
def data():
    df = pd.read_csv('./data/census.csv')
    return df

@pytest.fixture
def cat_features():
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    return cat_features
    

@pytest.fixture
def train(data,cat_features):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder = encoder, lb = lb
    )

    return X_train,y_train,X_test,y_test

def test_categorical_features(cat_features,data):
    print(set(cat_features))
    print(data.columns)
    print([feature  for feature in cat_features])
    assert all([feature in data.columns for feature in cat_features]) == True



def test_data_shape(data):
    assert data.shape[0] > 10000
    assert data.shape[1] == 15



def test_inference(train):
    X_train,y_train,X_test,y_test = train
    model = train_model(X_train, y_train)
    y_pred = inference(model,X_test)
    assert len(y_pred) > 0
    assert y_pred.shape == y_test.shape



