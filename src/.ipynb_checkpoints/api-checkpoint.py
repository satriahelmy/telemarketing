from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing
from pandas import json_normalize

config_data = util.load_config()
# ohe_stasiun = util.pickle_load(config_data["ohe_stasiun_path"])
le_encoder = util.pickle_load(config_data["le_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class Items(BaseModel):
    age : int
    duration : int
    campaign : int
    pdays : int
    previous : int
    job : str
    marital : str 
    education : str
    default : str
    housing : str
    loan : str
    contact : str
    month : str
    day_of_week : str
    poutcome : str
    empvarrate : float
    conspriceidx : float
    consconfidx : float
    euribor3m : float
    nremployed : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: Items):    

    data = pd.DataFrame(dict(data),index=[0])

    # Check range data
    try:
        data_pipeline.check_data(data, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    data = preprocessing.convertPdaysGroup(data)
    
    data = preprocessing.convertAgeGroup(data)
    
    # Encoding stasiun
    data = preprocessing.cat_ohe_transform(data,config_data)

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)