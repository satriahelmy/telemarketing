import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set

def convertPdaysGroup(df):
    """
    Fungsi untuk mengubah nilai pdays menjadi pdays_group
    Parameter:
    - df <pandas dataframe> : dataframe
    
    return:
    - df <pandas dataframe> : dataframe setelah konversi pdays_group
    """
    #mengelompokkan
    bins = [0, 7, 14, 30]
    labels = ['1w', '2w', '2wmore']
    df['pdays_group'] = pd.cut(df['pdays'], bins=bins, labels=labels, include_lowest=False)
    df['pdays_group'] = df['pdays_group'].astype('O')

    # fillna as Not contacted
    df['pdays_group'].fillna('Not contacted', inplace=True)
    df['pdays_group'].value_counts()
    df.drop(columns=['pdays'], axis=1, inplace=True)
    
    return df

def convertAgeGroup(df): 
    """
    Fungsi untuk mengubah nilai age menjadi age_group
    Parameter:
    - df <pandas dataframe> : dataframe
    
    return:
    - df <pandas dataframe> : dataframe setelah dikelompokkan menjadi age_group
    """
    #membagi age menjadi bin
    bins = [16, 30, 40, 50, 60, 100]
    labels = ['30less', '31-40', '41-50', '51-60', '60more']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    #hapus kolom age eksisting
    df.drop(columns=['age'], axis=1, inplace=True)
    return df

def cat_ohe_fit(config_data):
    for cat in config_data['predictors_categorical']:
        ohe_obj = OneHotEncoder(sparse = False,handle_unknown = 'ignore')
        # Fit ohe
        ohe_obj.fit(np.array(config_data["range_"+cat]).reshape(-1, 1))
        # Save ohe object
        util.pickle_dump(ohe_obj, config_data["ohe_"+cat+"_path"])
        
def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_path: str) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()
    
    # Load ohe stasiun
    ohe_obj = util.pickle_load(ohe_path)
    
    # Transform variable of set data, resulting array
    features = ohe_obj.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    column_name = [tranformed_column+"_"+s for s in ohe_obj.categories_[0]]
    features = pd.DataFrame(features.tolist(), columns = list(column_name))
    
    # Set index by original set data index
    features.set_index(set_data.index, inplace = True)
    
    # Concatenate new features with original set data
    set_data = pd.concat([features, set_data], axis = 1)
    
    # Drop stasiun column
    set_data.drop(columns = tranformed_column, inplace = True)
    
    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data

def cat_ohe_transform(set_data, config_data):
    set_data = set_data.copy()
    for cat in config_data['predictors_categorical']:
        set_data = ohe_transform(set_data,cat,config_data["ohe_"+cat+"_path"])
    return set_data

def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 26)

    # Balancing set data
    x_rus, y_rus = rus.fit_resample(set_data.drop("y", axis = 1), set_data['y'])

    # Concatenate balanced data
    set_data_rus = pd.concat([x_rus, y_rus], axis = 1)

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 11)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(set_data.drop("y", axis = 1), set_data["y"])

    # Concatenate balanced data
    set_data_ros = pd.concat([x_ros, y_ros], axis = 1)

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(set_data.drop("y", axis = 1), set_data["y"])

    # Concatenate balanced data
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)

    # Return balanced data
    return set_data_sm

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(le_encoder, le_path)

    # Return trained le
    return le_encoder

def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()

    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    
    # Return transformed label data
    return label_data

def join_data_train(train_set_rus, train_set_ros, train_set_sm):
    x_train = {
        "Undersampling" : train_set_rus.drop(columns = "y"),
        "Oversampling" : train_set_ros.drop(columns = "y"),
        "SMOTE" : train_set_sm.drop(columns = "y")
    }

    y_train = {
        "Undersampling" : train_set_rus.y,
        "Oversampling" : train_set_ros.y,
        "SMOTE" : train_set_sm.y
    }
    
    return x_train, y_train

def dump_data(x_train, y_train, valid_set, test_set):
    util.pickle_dump(x_train, "data/processed/x_train_feng.pkl")
    util.pickle_dump(y_train, "data/processed/y_train_feng.pkl")

    util.pickle_dump(valid_set.drop(columns = "y"), "data/processed/x_valid_feng.pkl")
    util.pickle_dump(valid_set.y, "data/processed/y_valid_feng.pkl")

    util.pickle_dump(test_set.drop(columns = "y"), "data/processed/x_test_feng.pkl")
    util.pickle_dump(test_set.y, "data/processed/y_test_feng.pkl")
    
    
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)
    
    # 3. Handling Pdays
    train_set = convertPdaysGroup(train_set)
    valid_set = convertPdaysGroup(valid_set)
    test_set = convertPdaysGroup(test_set)
    
    # 4. Handling Age
    train_set = convertAgeGroup(train_set)
    valid_set = convertAgeGroup(valid_set)
    test_set = convertAgeGroup(test_set)
    
    # 5. Fit OHE
    cat_ohe_fit(config_data)
    
    # 6. Transform OHE
    train_set = cat_ohe_transform(train_set,config_data)
    valid_set = cat_ohe_transform(valid_set,config_data)
    test_set = cat_ohe_transform(test_set,config_data)
    
    # 7. Undersampling
    train_set_rus = rus_fit_resample(train_set)
    
    # 8. Oversampling
    train_set_ros = ros_fit_resample(train_set)
    
    # 9. SMOTE
    train_set_sm = sm_fit_resample(train_set)
    
    # 10. Fit Label Encoding
    le_encoder = le_fit(config_data["label_categories"], config_data["le_path"])
    
    # 11. Transform Label Encoding
    train_set_rus.y = le_transform(train_set_rus.y, config_data)
    train_set_ros.y = le_transform(train_set_ros.y, config_data)
    train_set_sm.y = le_transform(train_set_sm.y, config_data)
    valid_set.y = le_transform(valid_set.y, config_data)
    test_set.y = le_transform(test_set.y, config_data)
    
    # 12. Join Data Train
    x_train, y_train = join_data_train(train_set_rus, train_set_ros, train_set_sm)
    
    # 13. Dump data
    dump_data(x_train, y_train, valid_set, test_set)
    
    
