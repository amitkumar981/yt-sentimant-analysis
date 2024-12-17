#importing libraries
import numpy as np
import pandas as pd
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import re

#logging configuration
logger=logging.getLogger('data_prerocessing')
logger.setLevel('DEBUG')

#add and configure console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#add and configure file handler
file_handler=logging.FileHandler('processing_error.log')
file_handler.setLevel('DEBUG')

#add and congfigure formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#add formatter to file_handler and console_handler
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

#add logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing Transformations on dataset"""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error('error in preprocessing of comment',e)
        raise
def normalize_text(df):
    try:
        df['clean_comment']=df['clean_comment'].apply(preprocess_comment)
        logger.debug('text normalization completed')
        return df
    except Exception as e:
        logger.error(f"error while normalization:{e}")

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) -> None:
    """Save processed train and test dataframe into root directory"""
    try:
        logger.debug('intializing saving data')
        interim_data_path=os.path.join(data_path,'interim')
        #creating root directory for saving dataframe
        os.makedirs(interim_data_path,exist_ok=True)
        logger.debug('creating directory completed')

    #saving dataframe
        logger.debug('intializing saving datalframe')
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        logger.debug('saving data_frame completed successfully')
    except Exception as e:
        logger.error(f"error in saving dataframe:{e}")
        raise
def main():
    try:

        logger.debug('intializing processing data')

        #fatch the data
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug('loading data successfully')

        #apply preprocessing
        train_processed_data=normalize_text(train_data)
        test_processed_data=normalize_text(test_data)
        logger.debug('preprocessing completed successfully')

        #save data
        save_data(train_processed_data,test_processed_data,'./data')
        logger.debug('saving data successfully')
    except Exception as e:
        logger.error('failed to complete preprocessing process:%s',e)

if __name__=='__main__':
    main()




    



    