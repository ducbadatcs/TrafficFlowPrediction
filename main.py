from typing import Any
import numpy as np
from process import windows
from train import get_gru, get_lstm, NUM_WINDOW, WINDOW_LENGTH, NUM_LOCATION, HORIZON, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

lstm = get_lstm(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)
gru = get_gru(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)

def predict_future(model, last_window, num_steps_ahead, step_size=24):
    current_window = last_window.copy()  # shape (1, WINDOW_LENGTH, 137)
    all_predictions = []
    
    while len(all_predictions) < num_steps_ahead:
        pred = model.predict(current_window, verbose=0)  # (1, 24, 137)
        all_predictions.append(pred[0])  # (24, 137)
        
        # Shift: remove first 24 timesteps, add last 24 predictions
        current_window = np.concatenate([
            current_window[:, step_size:, :],  # Remove first 24
            pred  # Add predictions
        ], axis=1)
    
    return np.vstack(all_predictions)[:num_steps_ahead]




# Your model + data should already be imported here.
# from your_module import predict_future, lstm, windows

st.title("Traffic Flow Prediction")

NUM_LOCATIONS = 137
BASE_TIME = datetime(2006, 10, 1, 0, 0)
DELTA_MINUTES = 15

def timestamp_to_datetime(ts):
    return BASE_TIME + timedelta(minutes=ts * DELTA_MINUTES)

location = st.sidebar.number_input("Location ID", min_value=0, max_value=NUM_LOCATIONS - 1, value=0)
num_steps = st.sidebar.slider("Number of prediction timestamps", min_value=1, max_value=100, value=24)

if st.sidebar.button("Predict"):
    # Run model prediction
    result = predict_future(lstm, windows[-1:], num_steps)
    predictions = [r[location] for r in result]

    start_ts = len(windows) * 24
    times = [timestamp_to_datetime(start_ts + i * 15) for i in range(num_steps)]

    st.subheader(f"Predictions for Location {location}")
    st.line_chart(predictions)

    st.write("Predictions:")
    for t, pred in zip(times, predictions):
        st.write(f"{t}: {pred:.2f}")
