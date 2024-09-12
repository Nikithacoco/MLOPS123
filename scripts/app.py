import streamlit as st
import pandas as pd
import pickle
import requests
import os
from PIL import Image
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename='logs/logfile_UI.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load the pre-fitted pipeline and model
def load_artifact(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        raise  # Re-raise the exception for handling in the calling code

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Adhoc Risk Profiling", "Batch Profiling"])

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Layout: Image on the left, title on the right
#col1, col2 = st.columns([1, 3])
#with col1:
    #image_path = os.path.join(script_dir, 'risk-image2.jfif')
    #image = Image.open(image_path)
    #st.image(image, use_column_width=True)

#with col2:
    #st.title("Financial Risk Assessment")
    #image_path = os.path.join(script_dir, 'risk-image.png')
    #image = Image.open(image_path)

# Navigation logic
if page == "Home":
    st.write("Welcome to the Financial Risk Assessment App.")
    st.write("Use the sidebar to navigate to Adhoc or Batch Profiling.")

elif page == "Adhoc Risk Profiling":
    st.header("Enter customer details:")
    
    # Adapting input fields to match financial risk dataset columns
    Age = st.number_input("Age", min_value=18, max_value=100)
    Gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary'])
    Education_Level = st.selectbox("Education Level", ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
    Marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widowed'])
    Income = st.number_input("Income", min_value=0)
    Credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    Loan_amount = st.number_input("Loan Amount", min_value=0)
    Loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Auto', 'Personal', 'Business'])
    Employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed'])
    Years_at_Current_Job = st.number_input("Years at Current Job", min_value=0)
    Payment_history = st.selectbox("Payment History", ['Excellent', 'Good', 'Fair', 'Poor'])
    Debt_to_Income_Ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0)
    Assets_value = st.number_input("Assets Value", min_value=0)
    Number_of_Dependents = st.number_input("Number of Dependents", min_value=0)
    previous_defaults = st.number_input("Previous Defaults", min_value=0)
    marital_status_change = st.number_input("Marital Status Change", min_value=1, max_value=2)  # Ensure valid range
    #Risk_Rating = st.number_input("Risk Rating", ['Low','Medium','High'] )
  
    # New location-based fields
    # city = st.text_input("City")
    # state = st.text_input("State")
    # country = st.text_input("Country")

    
    # Predict button
    if st.button('Predict Risk'):
        pipeline = load_artifact(os.path.join(script_dir, 'data_processing_pipeline.pkl'))
        model = load_artifact(os.path.join(script_dir, 'best_classifier.pkl'))
        label_encoder = load_artifact(os.path.join(script_dir, 'label_encoder.pkl'))

        input_df = pd.DataFrame([[Age, Gender, Education_Level, Marital_status, Income, Credit_score, Loan_amount, Loan_purpose, Employment_status, Years_at_Current_Job, Payment_history, Debt_to_Income_Ratio, Assets_value, Number_of_Dependents, previous_defaults, marital_status_change]],
            columns=['Age', 'Gender', 'Education Level', 'Marital Status', 'Income', 'Credit Score', 'Loan Amount', 'Loan Purpose', 'Employment Status', 'Years at Current Job', 'Payment History', 'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults', 'Marital Status Change'])

        
        logging.info(f"User input data frame created")
        
        # Use the pre-fitted pipeline to transform the input data
        transformed_input = pipeline.transform(input_df)
        logging.info(f"User input data frame is transformed")

        # Make prediction with the preloaded model
        # prediction = model.predict(transformed_input)
        # logging.info(f"Received Prediction: {prediction}")

        # # Display the prediction
        # st.subheader('Predicted Risk Rating:')
        # st.write(prediction[0])
        try:
            # Transform the input data and make predictions
            transformed_input = pipeline.transform(input_df)
            prediction = model.predict(transformed_input)
            decoded_prediction = label_encoder.inverse_transform(prediction)

            # Display the predicted category
            st.subheader('Predicted Risk Category:')
            st.write(decoded_prediction[0])
            logging.info(f"Prediction: {decoded_prediction[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            logging.error(f"Error during prediction: {str(e)}")

elif page == "Batch Profiling":
    st.header("Batch Profiling")
    uploaded_file = st.file_uploader("Upload your CSV file for batch prediction", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logging.info(f"Batch file uploaded with {len(df)} records")
        
        try:
            response = requests.post("http://localhost:8001/batch_predict", json={"data": df.to_dict(orient="list")})

            response.raise_for_status()
            predictions = response.json()
            output_df = pd.DataFrame(predictions)
            output_file_path = os.path.join(script_dir, 'batch_predictions.csv')
            output_df.to_csv(output_file_path, index=False)
            st.success(f"Batch predictions saved to {output_file_path}")
            logging.info(f"Batch predictions saved to {output_file_path}")
        except requests.exceptions.RequestException as req_err:
            st.error("Error during batch prediction. Please check the API service.")
            logging.error(f"Batch prediction failed: {req_err}")
