"""import pandas as pd
import logging

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    df = pd.read_csv('C:/Users/HP/Desktop/financial_risk_assessment.csv', sep=';', header=0)
    logging.info('Data loaded successfully.')

    # Print the first few rows to check the data
    print("First few rows of the DataFrame:")
    print(df.head())

    # Print the columns to verify names
    print("Initial DataFrame columns:")
    print(df.columns.tolist())

    # Strip any leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Verify cleaned column names
    print("Cleaned DataFrame columns:")
    print(df.columns.tolist())

    # Check if 'Risk Rating' column is present
    if 'Risk Rating' in df.columns:
        # Feature engineering
        X = df.drop('Risk Rating', axis=1)
        Risk_Rating = df['Risk Rating']

        # Encode response variable (assuming encode_response_variable is defined)
        Risk_Rating_encoded = encode_response_variable(Risk_Rating)

        # Create and fit the data processing pipeline
        pipeline = create_data_pipeline(X)
        pipeline.fit(X)
        logging.info('Data processing pipeline created and fitted.')

        # Save the pipeline
        save_pipeline(pipeline, 'data_processing_pipeline.pkl')
        logging.info('Data processing pipeline saved.')

        # Transform the data
        X_transformed = pipeline.transform(X)

        # Split the data
        X_train, X_val, y_train, y_val = split_data(X_transformed, Risk_Rating_encoded)

        # Train the model
        best_model = training_pipeline(X_train, y_train)

        # Make predictions
        predictions = prediction_pipeline(X_val)

        # Evaluate the model
        conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val)

        logging.info('Model training, prediction, and evaluation completed.')
    else:
        logging.error("Column 'Risk Rating' not found in the DataFrame.")
        print("Column 'Risk Rating' is not found in the DataFrame.")

if __name__ == "__main__":
  main()"""

import pandas as pd
from data_preprocessing import create_data_pipeline, save_pipeline, load_pipeline, split_data, encode_response_variable
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices
from helper_functions import logging


def main():
    # Configure logging (optional, adjust log level and output destination as needed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data/Banking_Credit_Risk_Data.csv
    df = pd.read_csv('C:/Users/HP/Desktop/financial_risk_assessment.csv')
    logging.info('Data loaded successfully.')

    # Feature engineering (replace with your feature engineering steps)
    X = df.drop(['Risk Rating'], axis=1)
    y = df['Risk Rating']

    # Encode response variable (assuming encode_response_variable is defined)
    y_encoded = encode_response_variable(y)

    # Create and fit the data processing pipeline (replace with create_data_pipeline)
    pipeline = create_data_pipeline(X)
    pipeline.fit(X)
    logging.info('Data processing pipeline created and fitted.')

    # Save the pipeline for later use (assuming save_pipeline is defined)
    save_pipeline(pipeline, 'data_processing_pipeline.pkl')
    logging.info('Data processing pipeline saved.')

    # Transform the data using the fit_transform method
    X_transformed = pipeline.transform(X)
    
    # X_transformed = X_transformed.toarray()
    # Split the data for training and validation
    X_train, X_val, y_train, y_val = split_data(X_transformed, y_encoded)

    # Train the best model (replace with training_pipeline)
    best_model = training_pipeline(X_train, y_train)

    # Make predictions (replace with prediction_pipeline)
    predictions = prediction_pipeline(X_val)

    # Evaluate the model (replace with evaluation_matrices)
    conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val)

    logging.info('Model training, prediction, and evaluation completed.')


if __name__ == "__main__":
    main()