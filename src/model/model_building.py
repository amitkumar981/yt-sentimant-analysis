#importing libraries 
import numpy as  np
import pandas as pd
from lightgbm import LGBMClassifier
import pickle
import yaml
import os
import logging

#add and configure logger
logger=logging.Logger('model_buliding')
logger.setLevel('DEBUG')

#add and configure console hanlder
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#add and configure file handler
file_hanlder=logging.FileHandler('model_building_error.log')
file_hanlder.setLevel('DEBUG')

#add formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_hanlder.setFormatter(formatter)

#add logger to file and console_hanlder
logger.addHandler(console_handler)
logger.addHandler(file_hanlder)

# create a function to load parametters
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

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
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
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        with open(os.path.join(root_dir, 'data/processed/x_train_tfidf.pkl'), 'rb') as f:
            x_train_tfidf, y_train = pickle.load(f)
            logger.debug('TF-IDF data loaded for model training')
            
              # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lgbm(x_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

        
if __name__ == '__main__':
    main()




