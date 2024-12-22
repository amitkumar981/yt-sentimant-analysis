import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import os
import pickle
import logging

# Add and configure logger
logger = logging.Logger('feature_engineering')
logger.setLevel('DEBUG')

# Add and configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Add and configure file handler
file_handler = logging.FileHandler('feature_engineering_error.log')
file_handler.setLevel('DEBUG')

# Add formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function to load parameters from yaml file
def load_params(param_path: str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(param_path) as file:
            params = yaml.safe_load(file)
            logger.debug('params retrieved from: %s', param_path)
            return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', param_path, e)
        raise

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    """Load csv file from source"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('data loaded from: %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

# Function to apply vectorization
def apply_tfidf(train_data: pd.DataFrame, ngram_range: tuple, max_features: int) -> tuple:
    """Apply vectorization"""
    try:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        x_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        x_train_tfidf = vectorizer.fit_transform(x_train)
        logger.debug(f"vectorization complete with shape {x_train_tfidf.shape}")

        # Save vectorizer into root directory
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
            logger.debug('TF-IDF vectorizer saved')
        return x_train_tfidf, y_train
    except Exception as e:
        logger.error('Error applying TF-IDF vectorization: %s', e)
        raise

def ensure_directory_exists(directory: str):
    """Ensure that a directory exists, create if not."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug('Created directory: %s', directory)
    except Exception as e:
        logger.error('Error ensuring directory exists %s: %s', directory, e)
        raise

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, '../../'))
    except Exception as e:
        logger.error('Error getting root directory: %s', e)
        raise

def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply TF-IDF feature engineering on training data
        x_train_tfidf, y_train = apply_tfidf(train_data, ngram_range, max_features)

        # Ensure the processed data directory exists
        processed_data_dir = os.path.join(root_dir, 'data/processed')
        ensure_directory_exists(processed_data_dir)

        # Save the transformed data
        with open(os.path.join(processed_data_dir, 'x_train_tfidf.pkl'), 'wb') as f:
            pickle.dump((x_train_tfidf, y_train), f)
            logger.debug('TF-IDF applied and data saved')
    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise

if __name__ == '__main__':
    main()







    
        
    








    
