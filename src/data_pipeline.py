from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np 
import joblib
import os
import yaml
import util as util

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i,delimiter=';'), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def removeDuplicates(data):
    
     # Drop duplicate
    data = data.drop_duplicates()
    return data

def handlingPdays(data):
    data['pdays'] = data['pdays'].replace(999, -1)
    return data

def renameColumn(data):
    data.columns = data.columns.str.replace(".", "")
    return data

def check_data(input_data, params):
    # Check data types
    assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "an error occurs in float64 column(s)."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int64_columns"], "an error occurs in int64 column(s)."

    # Check range of data
    assert input_data["age"].between(params["range_age"][0], params["range_age"][1]).sum() == len(input_data), "an error occurs in age range."
    assert input_data["duration"].between(params["range_duration"][0], params["range_duration"][1]).sum() == len(input_data), "an error occurs in duration range."
    assert input_data["campaign"].between(params["range_campaign"][0], params["range_campaign"][1]).sum() == len(input_data), "an error occurs in campaign range."
    assert input_data["pdays"].between(params["range_pdays"][0], params["range_pdays"][1]).sum() == len(input_data), "an error occurs in pdays range."
    assert input_data["previous"].between(params["range_previous"][0], params["range_previous"][1]).sum() == len(input_data), "an error occurs in previous range."
    assert input_data["empvarrate"].between(params["range_empvarrate"][0], params["range_empvarrate"][1]).sum() == len(input_data), "an error occurs in emp.var.rate range."
    assert input_data["conspriceidx"].between(params["range_conspriceidx"][0], params["range_conspriceidx"][1]).sum() == len(input_data), "an error occurs in cons.price.idx range."
    assert input_data["consconfidx"].between(params["range_consconfidx"][0], params["range_consconfidx"][1]).sum() == len(input_data), "an error occurs in cons.conf.idx range."
    assert input_data["euribor3m"].between(params["range_euribor3m"][0], params["range_euribor3m"][1]).sum() == len(input_data), "an error occurs in euribor3m range."
    assert input_data["nremployed"].between(params["range_nremployed"][0], params["range_nremployed"][1]).sum() == len(input_data), "an error occurs in nr.employed range."
    assert set(input_data["job"]).issubset(set(params["range_job"])), "an error occurs in job range."
    assert set(input_data["marital"]).issubset(set(params["range_marital"])), "an error occurs in marital range."
    assert set(input_data["education"]).issubset(set(params["range_education"])), "an error occurs in education range."
    assert set(input_data["default"]).issubset(set(params["range_default"])), "an error occurs in default range."
    assert set(input_data["housing"]).issubset(set(params["range_housing"])), "an error occurs in housing range."
    assert set(input_data["loan"]).issubset(set(params["range_loan"])), "an error occurs in loan range."
    assert set(input_data["contact"]).issubset(set(params["range_contact"])), "an error occurs in contact range."
    assert set(input_data["month"]).issubset(set(params["range_month"])), "an error occurs in month range."
    assert set(input_data["day_of_week"]).issubset(set(params["range_day_of_week"])), "an error occurs in day_of_week range."
    assert set(input_data["poutcome"]).issubset(set(params["range_poutcome"])), "an error occurs in poutcome range."
    
    
def splitInputOtput(data,config_data):
    x = data[config_data["predictors"]].copy()
    y = data[config_data["label"]].copy()
    return x,y

def splitTrainTest(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)
    return x_train, x_test, y_train, y_test

def splitValidTest(x_test,y_test):
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)
    return x_valid, x_test, y_valid, y_test

def dumpData(x_train, y_train, x_valid, y_valid, x_test, y_test, config_data):
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
    
    
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()
    
    # 2. Read raw dataset
    raw_dataset = read_raw_data(config_data)
    raw_dataset = renameColumn(raw_dataset)
    
    # 3. Remove duplicates data
    raw_dataset = removeDuplicates(raw_dataset)
    
    # 4. Handling pdays column
    raw_dataset = handlingPdays(raw_dataset)
    
    # 5. Check Data
    check_data(raw_dataset,config_data)
    
    # 6. Split input output
    x,y = splitInputOtput(raw_dataset, config_data)
    
    # 7. Split Train Test
    x_train, x_test, y_train, y_test = splitTrainTest(x,y)
    
    #8. Split Valid dan Test
    x_valid, x_test, y_valid, y_test = splitValidTest(x_test,y_test)
    
    #9. Dump data to pickled
    dumpData(x_train, y_train, x_valid, y_valid, x_test, y_test, config_data)