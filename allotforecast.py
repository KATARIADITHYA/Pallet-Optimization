import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from streamlit.components.v1 import iframe

# Load the pre-trained model
model=pickle.load(open("/Users/adithyakatari/Documents/pallet optimization project/streamlit files/allotforecast_2024.pkl",'rb'))



# Define the function for predicting the label
def predict_label(input_data):
    p=int(input_data)
    d1=pd.DataFrame()
    for i in range(155,155+p):
        d1 = pd.concat([d1, pd.DataFrame(model.predict(i))])
    return d1


# Define the Streamlit app
def main():
    st.title('Forecasting quantity')
    image_url = 'https://img.freepik.com/free-vector/warehouse-with-man-worker-forklift-boxes_107791-7383.jpg?w=2000'
    st.image(image_url, caption="My Image", use_column_width=True)
    # Define the input form for the user
    input_data = st.text_input('Enter an value to predict')
    # Make a prediction when the user clicks the "Predict" button
    if st.button('Predict'):
        try:
            value = int(input_data)
            if value <= 0:
                st.write("Invalid input. Please enter a positive integer value.")
            else:
                prediction = predict_label(input_data)
                # Display the predicted label to the user
                st.write('Prediction:', prediction)
        except ValueError:
            st.write("Invalid input. Please enter an positive integer value.")

if __name__ == '__main__':
    main()
