from typing import Any
import numpy as np
import gradio as gr
import pandas as pd
from matplotlib import pyplot as plt
from process import windows, scats
from sklearn.preprocessing import StandardScaler
from train import get_gru, get_lstm, NUM_WINDOW, WINDOW_LENGTH, NUM_LOCATION, HORIZON, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH
from train import normalize, denormalize, scaler
import numpy as np
from datetime import datetime, timedelta

lstm = get_lstm(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)
gru = get_gru(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)

SCATS_TO_LOC = scats
def predict_future(model: Any, last_window: np.ndarray, scaler: StandardScaler, num_steps_ahead: int, step_size=24):
    """
    Predict future traffic values with normalization handled automatically.
    
    Args:
        model: Trained keras model.
        last_window: np.ndarray of shape (1, WINDOW_LENGTH, NUM_LOCATION)
        scaler: sklearn StandardScaler fitted on training data.
        num_steps_ahead: Total number of timesteps to predict.
        step_size: Number of timesteps to shift window each iteration.
    
    Returns:
        np.ndarray of shape (num_steps_ahead, NUM_LOCATION) with denormalized predictionsa.
    """
    # Normalize input
    current_window = normalize(scaler, last_window)
    all_predictions = []

    i = 0
    while len(all_predictions) < num_steps_ahead:
        i += 1
        pred_n = model.predict(current_window, verbose=0)  # normalized predictions
        all_predictions.append(pred_n[0])
        # print(f"Prediction #{i}: {pred_n[0]}")
        current_window = np.concatenate([
            current_window[:, step_size:, :],
            pred_n
        ], axis=1)

    predictions_n = np.vstack(all_predictions)[:num_steps_ahead]

    # Denormalize output
    predictions = denormalize(scaler, predictions_n)
    return predictions


def predict_interval(model: Any, last_window: np.ndarray, scaler: StandardScaler, 
                     scats_number: str, start_time_str: str, end_time_str: str) -> np.ndarray:
    """
    Predict for a given SCATS number and datetime interval.
    
    Args:
        model: trained keras model
        last_window: last observed window (1, WINDOW_LENGTH, NUM_LOCATION)
        scaler: fitted StandardScaler
        scats_number: str, e.g., '3002'
        start_time_str: 'YYYY-MM-DD HH:MM'
        end_time_str: 'YYYY-MM-DD HH:MM'
    
    Returns:
        np.ndarray of shape (num_steps_in_interval, num_locations_for_SCATS)
    """
    loc_indices = SCATS_TO_LOC.get(scats_number)
    if not loc_indices:
        raise ValueError("SCATS number not found!")

    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)
    # should predict from here
    ref_time = pd.Timestamp("2006-11-01 00:00")  # reference start

    start_step = int((start_time - ref_time) / pd.Timedelta(minutes=15))
    end_step = int((end_time - ref_time) / pd.Timedelta(minutes=15))

    total_steps_needed = end_step + 1

    # Predict
    predictions = predict_future(model, last_window, scaler, total_steps_needed)

    # Slice interval and select SCATS locations
    pred_interval = predictions[start_step:end_step + 1, loc_indices]
    # should be already flattened? Just to be sure.
    return np.array(np.median(pred_interval, axis=1)).flatten()

# ...existing code...
def predict_traffic(scats_number: str, start_time_str: str, end_time_str: str, model_choice="GRU"):
    try:
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
        if start_time.minute % 15 or end_time.minute % 15:
            raise ValueError("Timestamps must align to 15-minute intervals.")
        if start_time > end_time:
            raise ValueError("Start time must be before end time.")

        model = gru if model_choice == "GRU" else lstm
        last_window = normalize(scaler, windows[-1:])
        pred_interval = predict_interval(model, last_window, scaler, scats_number, start_time_str, end_time_str)

        timestamps = pd.date_range(start=start_time, end=end_time, freq="15min")
        if len(timestamps) != len(pred_interval):
            raise ValueError(f"Timestamp count {len(timestamps)} does not match prediction length {len(pred_interval)}.")

        text = (
            f"Predicted flow for SCATS {scats_number}\n"
            f"From {start_time} to {end_time}\n"
            f"{len(pred_interval)} intervals (15 min)\n\n"
            f"{pred_interval}"
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(timestamps, pred_interval, marker="o")
        ax.set_title(f"Predicted Flow for SCATS {scats_number}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flow")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        return text, fig

    except Exception as e:
        return f"Error: {str(e)}", None


if __name__ == "__main__":
    # Create Gradio interface
    interface = gr.Interface(
    fn=predict_traffic,
    inputs=[
        gr.Textbox(label="SCATS Number", placeholder="3002"),
        gr.Textbox(label="Start Time", placeholder="2006-11-05 09:30"),
        gr.Textbox(label="End Time", placeholder="2006-11-05 11:30"),
        gr.Radio(["LSTM", "GRU"], label="Model", value="GRU")
    ],
    outputs=[
        gr.Textbox(label="Prediction Output"),
        gr.Plot(label="Flow Plot")
    ],
    title="Traffic Flow Prediction",
    description="Predict traffic flow for a given SCATS location and time interval")

interface.launch(share=True)