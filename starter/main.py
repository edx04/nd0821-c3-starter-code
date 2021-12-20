# Put the code for your API here.
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel,Field
import pandas as pd 
import json
import logging
from starter.ml.data import *
from starter.ml.model import *
import joblib

logging.basicConfig( level=logging.DEBUG)


class modelParam(BaseModel):
    age : int 
    workclass : str 
    fnlgt : int 
    education : str 
    education_num : int = Field(alias='education-num')
    marital_status : str = Field(alias='marital-status')
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int = Field(alias='capital-gain')
    capital_loss : int = Field(alias='capital-loss')
    hours_per_week : int = Field(alias='hours-per-week')
    native_country : str = Field(alias='native-country')



    class Config:
        schema_extra = {
            "example": {
                'age': 34,
                'workclass': 'Private',
                'fnlgt': 245487,
                'education': '7th-8th',
                'education-num': 4,
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Transport-moving',
                'relationship': 'Husband',
                'race': 'Amer-Indian-Eskimo',
                'sex': 'Male',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 45,
                'native-country': 'Mexico'
            }
        }



app = FastAPI()

@app.get("/")
async def welcome():
    return {"response":"welcome"}

@app.post("/predict")
async def predict(params:modelParam):
    #df = pd.read_json(model)

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


    model = joblib.load("./model/model.pkl")
    encoder = joblib.load("./model/encoder.pkl")
    lb = joblib.load("./model/lb.pkl")

    df = pd.DataFrame(jsonable_encoder(params),index=[0])
    data, _, _, _ = process_data(
        df, categorical_features=cat_features, encoder=encoder, training=False
    )
    y_pred = inference(model,data)

    logging.info(y_pred[0])
    return {"prediction": "<=50k" if y_pred == 0 else ">50k" }
