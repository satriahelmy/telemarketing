import preprocessing
import util as utils
import pandas as pd
import numpy as np

def test_convert_pdays():
    # Arrange
    config = utils.load_config()

    mock_data = {
        "pdays" : [-1,1,5,10,13,20]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        "pdays_group" : [
            "Not contacted", "1w", "1w", "2w", "2w", "2wmore"]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.convertPdaysGroup(mock_data)

    # Assert
    assert processed_data.equals(expected_data)
    
def test_convert_age():
    # Arrange
    config = utils.load_config()

    mock_data = {
        "age" : [29,35,43,55,67]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        "age_group" : ['30less', '31-40', '41-50', '51-60', '60more']}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.convertAgeGroup(mock_data)

    # Assert
    print(processed_data)
    

def test_le_transform():
    # Arrange
    config = utils.load_config()
    mock_data = {"y" : ["yes", "no", "yes", "yes", "no", "yes"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"y" : [1, 0, 1, 1, 0, 1]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(int)

    # Act
    processed_data = preprocessing.le_transform(mock_data["y"], config)
    processed_data = pd.DataFrame({"y" : processed_data})

    # Assert
    assert processed_data.equals(expected_data)