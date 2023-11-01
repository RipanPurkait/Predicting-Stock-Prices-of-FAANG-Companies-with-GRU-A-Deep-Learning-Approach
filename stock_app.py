import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Streamlit app title and description
st.title("Stock Price Prediction Web App")
st.write("Predicted stock prices based on a pre-trained model.")

# Load the pre-trained model
model_path = r"D:\Streamlit\Stock_market_prediction\GRU.h5"  # Replace with the path to your .h5 model file
model = load_model(model_path)

# Get user input

stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")

# Fetch historical stock data
if stock_symbol:
    start_date = '2010-08-20'
    end_date = datetime.now().date()

    # Fetch the data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Store the fetched data as a pandas DataFrame
    df = pd.DataFrame(data)

    # Visualization
    st.title("Stock Closing Prices Over Time")
    fig, ax = plt.subplots(figsize=(20, 8))
    df['Close'].plot(ax=ax)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Closing Prices Over Time')
    st.pyplot(fig, use_container_width=True)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(df[['Close']])

    sequence_length = 10  # You can adjust this as needed

    X = []
    y = []

    # Create sequences
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    split_ratio = 0.8  # 80% for training, 20% for testing
    split_index = int(len(scaled_data) * split_ratio)

    X_train = scaled_data[:split_index - sequence_length]
    y_train = scaled_data[sequence_length:split_index]

    X_test = scaled_data[split_index - sequence_length:-sequence_length]
    y_test = scaled_data[split_index:]


    loss = model.evaluate(X_test, y_test)
    # print("Test loss:", loss)

    # Make predictions
    predictions = model.predict(X_test).reshape(-1,1)

    st.title("Actual vs Predicted")
    # Denormalize predictions
    predictions = scaler.inverse_transform(predictions)
    y_test_denorm = scaler.inverse_transform(y_test)

    how_actual = True
    show_predicted = True

    # Plotting
    fig, ax = plt.subplots(figsize=(25, 10))
    if how_actual:
        ax.plot(y_test_denorm, label='Actual', color='blue')
    if show_predicted:
        ax.plot(predictions, label='Predicted', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig) 
    
    

    sequence_length = 10

    # Assuming you have a trained LSTM model named 'model' and a scaler named 'scaler'

    # Take the last sequence_length data points from the test set
    last_sequence = X_test[-sequence_length:]

    # Create a list to store the predicted values
    predicted_values = []

    # Predict the next 10 days iteratively
    for _ in range(10):
        # Reshape the last_sequence to match the input shape of the model
        last_sequence_reshaped = last_sequence.reshape(-1,1)

        # Predict the next value
        next_prediction = model.predict(last_sequence_reshaped)

        # Append the predicted value to the list
        predicted_values.append(next_prediction[0, 0])

        # Update the last_sequence for the next iteration
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction[0, 0]  # Update the last element

    # Denormalize the predicted values
    predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))



    # Create an array of days for the next 10 days
    next_10_days = np.arange(1, 11)

    # Create a Streamlit app
    st.title("Predicted Values for the Next 10 Days")

    # Plot the predicted values using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(next_10_days, predicted_values, marker='o', linestyle='-')
    ax.set_title("Predicted Values for the Next 10 Days")
    ax.set_xlabel("Day")
    ax.set_ylabel("Predicted Value")
    ax.grid(True)

    # Display the plot in the Streamlit app
    st.pyplot(fig)


    