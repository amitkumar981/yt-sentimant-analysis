#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
import os

#logging configuration

#create and configure logger
logger = logging.getLogger('data_injection')
logger.setLevel(logging.DEBUG)

#create and configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

#create and configure file handler
file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.DEBUG)

#set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#adding formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

#create a function to load parameters from yaml file
def load_params(params_path: str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Params retrieved from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

#create a function to load dataset from url
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from url"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Retrieved data from: %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise

#create a function for basic preprocessing of data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        #drop missing values 
        df.dropna(inplace=True)
        #drop duplicates
        df.drop_duplicates(inplace=True)
        #remove white space 
        df = df[df['clean_comment'].str.strip() != '']

        #logging
        logger.debug('Basic preprocessing completed: removed missing values, duplicates, and whitespace from data')
        return df
    except KeyError as e:
        logger.error('Missing column in dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

#create a function to save data
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Save the train_df and test_df"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')

        #make directory if it doesn't exist
        os.makedirs(raw_data_path, exist_ok=True)

        #save data
        train_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        #logging
        logger.debug('Saved data successfully to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving: %s', e)
        raise

def main():
    try:
        #load parameters from param.yaml from root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_injection']['test_size']

        #load data from specific url
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        #preprocess the data
        final_df = preprocess_data(df)

        #splitting
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()







    
                  

    




    

   
    