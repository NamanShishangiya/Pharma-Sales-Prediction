from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st

data = pd.read_csv("C:/Users/NAMAN/Desktop/Rajkot_SML.csv")

data.dtypes

data.isnull().count()

data = data.dropna()

data = data.replace(',','',regex=True)

from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st

# Extract input and output features
X = data[['TOTAL(April)', 'TOTAL(May)', 'TOTAL(June)', 'TOTAL(July)', 'TOTAL(August)', 'TOTAL(September)']]
y = data['TOTAL(October)']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Define function to predict for a given month
def predict_for_month(month_data):
    # Create input for prediction
    X_pred = [[month_data['TOTAL(April)'], month_data['TOTAL(May)'], month_data['TOTAL(June)'], month_data['TOTAL(July)'], month_data['TOTAL(August)'], month_data['TOTAL(September)']]]

    # Use the model to make prediction
    y_pred = model.predict(X_pred)

    # Return the predicted value
    return y_pred[0]

# Create the Streamlit web app
st.title("Sales Prediction Model")

# Take input from the user for the month they want to predict
month_data = {}
try:
    month_data['TOTAL(April)'] = st.number_input('Enter total sales for April: ')
    month_data['TOTAL(May)'] = st.number_input('Enter total sales for May: ')
    month_data['TOTAL(June)'] = st.number_input('Enter total sales for June: ')
    month_data['TOTAL(July)'] = st.number_input('Enter total sales for July: ')
    month_data['TOTAL(August)'] = st.number_input('Enter total sales for August: ')
    month_data['TOTAL(September)'] = st.number_input('Enter total sales for September: ')
    month_name = st.text_input('Enter the month for which you want to predict sales (e.g. October): ')

    # Add the input data for the specified month
    month_data[f'TOTAL({month_name})'] = 0.0

    # Predict the total sales for the specified month
    total_sales = predict_for_month(month_data)

    st.write(f'Predicted total sales for {month_name}: {total_sales:.2f}')
except ValueError:
    st.write('Invalid input. Please enter numeric values for total sales.')
