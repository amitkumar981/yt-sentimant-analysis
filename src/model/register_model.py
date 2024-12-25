#importing libraries
import numpy as np
import json
import os
import mlflow
import logging


# Configure logger
logger = logging.getLogger('register_model')
logger.setLevel(logging.DEBUG)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Configure file handler
file_handler = logging.FileHandler('register_model_error.log')
file_handler.setLevel(logging.DEBUG)

# Configure formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Set MLflow server URI
mlflow.set_tracking_uri('http://13.238.159.116:5000/')

# Create a function to load model info
def load_model_info(file_path: str) -> dict:
    """Load model_info from model_info json file"""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
            logger.debug('Retrieved model_info successfully from %s', file_path)
            return model_info
    except Exception as e:
        logger.error('Error loading model_info from %s: %s', file_path, e)
        raise

# Create a function to register and add model into staging
def register_model(model_name: str, model_info: dict) -> None:
    """Register model to MLflow model registry"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.debug('Model registered with URI: %s and version: %s', model_uri, model_version.version)

        # Transition the model into staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=model_name,
                                              version=model_version.version,
                                              stage='Staging')
        logger.debug('Model version %s transitioned to staging', model_version.version)

    except Exception as e:
        logger.error('Error registering or transitioning model: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = 'yt_chrome_plugin_model'
        register_model(model_name, model_info)

    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise

if __name__ == '__main__':
    main()





    

        