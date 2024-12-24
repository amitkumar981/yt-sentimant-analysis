#import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import logging
import yaml
import json
import mlflow
from mlflow.models import infer_signature

# Configure logger
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Configure file handler
file_handler = logging.FileHandler('model_evaluation_error.log')
file_handler.setLevel(logging.DEBUG)

# Configure formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from source"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Retrieved data from: %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_param(param_path: str) -> dict:
    """Load parameters from params.yaml"""
    try:
        with open(param_path, 'r') as file:
            param = yaml.safe_load(file)
            logger.debug('Retrieved parameters from: %s', param_path)
            return param
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', param_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load vectorizer.pkl file"""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
            logger.debug('Retrieved vectorizer from %s', vectorizer_path)
            return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_model(model_path: str):
    """Load model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logger.debug('Retrieved model from: %s', model_path)
            return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def model_evaluation(model, x_test: np.array, y_test: np.array):
    """Evaluate model and log classification report and confusion matrix"""
    try:
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation complete')
        return cm, report
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion_matrix as artifact"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {dataset_name}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save the confusion matrix
        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        plt.close()
        mlflow.log_artifact(cm_file_path)
        logger.debug('Logged confusion matrix for %s', dataset_name)
    except Exception as e:
        logger.error('Error logging confusion matrix for %s: %s', dataset_name, e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model info"""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
            logger.debug('Model info saved to: %s', file_path)
    except Exception as e:
        logger.error('Error saving model info to %s: %s', file_path, e)
        raise

def main():
    try:
        # Set experiment URI
        mlflow.set_tracking_uri('http://13.238.159.116:5000/')

        # Set MLflow experiment
        mlflow.set_experiment('dvc_pipeline_run')

        with mlflow.start_run() as run:
            # Get root directory
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            logger.debug('Root directory: %s', root_dir)

            # Load parameters from yaml file
            params = load_param(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))

            # Log LightGBM parameters
            lgbm_params = model.get_params()
            for key, value in lgbm_params.items():
                mlflow.log_param(f"lgbm_{key}", value)

            # Load data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare data
            x_test = test_data['clean_comment'].values
            y_test = test_data['category'].values
            x_test_tfidf = vectorizer.transform(x_test)

            # Create an input example for signature
            input_example = pd.DataFrame(x_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())

            # Infer the signature
            signature = infer_signature(input_example, model.predict(x_test_tfidf[:5]))

            # Log the model with signature
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Save model info
            model_path = 'model'
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log vectorizer
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            cm, report = model_evaluation(model, x_test_tfidf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise

if __name__ == '__main__':
    main()













    




    
