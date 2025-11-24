"""
Train the NN model.
"""
from sklearn import metrics
import argparse
import keras
import numpy as np
import os
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed, RepeatVector
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.saving import load_model
from keras.utils import plot_model
from process import windows, make_dataset
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict
import pandas as pd

def fit_scaler(A: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(A.reshape(-1, 1))

def normalize(scaler: StandardScaler, A: np.ndarray) -> np.ndarray:
    return scaler.transform(A.reshape(-1, 1)).reshape(A.shape)

# for predictions
def denormalize(scaler: StandardScaler, A: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(A.reshape(-1, 1)).reshape(A.shape)

X_train, X_val, X_test, y_train, y_val, y_test = make_dataset(windows)
scaler = fit_scaler(X_train)
X_train_n, X_val_n, X_test_n = normalize(scaler, X_train), normalize(scaler, X_val), normalize(scaler, X_test)
y_train_n, y_val_n, y_test_n = normalize(scaler, y_train), normalize(scaler, y_val), normalize(scaler, y_test)

NUM_WINDOW, WINDOW_LENGTH, NUM_LOCATION = windows.shape
HORIZON = 24

INPUT_SEQUENCE_LENGTH = WINDOW_LENGTH - HORIZON # 648
OUTPUT_SEQUENCE_LENGTH = HORIZON # 24

def get_lstm(input_seq_len: int, output_seq_len: int, num_locations: int) -> Any:
    if os.path.exists("./model/lstm.keras"):
        return load_model("./model/lstm.keras")

    units = 256

    model = Sequential([
        Input(shape=(input_seq_len, num_locations)),

        # Encoder
        LSTM(units, return_sequences=False),
        Dropout(0.2),

        # Repeat context for decoder
        RepeatVector(output_seq_len),

        # Decoder
        LSTM(units, return_sequences=True),
        Dropout(0.1),

        # Output projection
        TimeDistributed(Dense(num_locations))
    ])
    
    return model

def get_gru(input_seq_len: int, output_seq_len: int, num_locations: int) -> Any:
    if os.path.exists("./model/gru.keras"):
        return load_model("./model/gru.keras")

    units = 256

    model = Sequential([
        Input(shape=(input_seq_len, num_locations)),

        # Encoder
        GRU(units, return_sequences=False),
        Dropout(0.2),

        # Repeat context for decoder
        RepeatVector(output_seq_len),

        # Decoder
        GRU(units, return_sequences=True),
        Dropout(0.1),

        # Output projection
        TimeDistributed(Dense(num_locations))
    ])
    
    return model

def eva_regress(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # vs = metrics.explained_variance_score(y_true, y_pred)
    # mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # print('explained_variance_score:%f' % vs)
    # print('mape:%f%%' % mape)
    print(f'mae: {mae}')
    print(f'mse: {mse}' % mse)
    print(f'rmse: {np.sqrt(mse)}')
    print(f'r2: {r2}')


def train_model(model: keras.Sequential, 
                X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                name: str, config: dict[str, Any] = {"epochs": 100, "batch_size": 32}):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    
    model.summary()
    print(f"Plotted to ./images/{name}.png")
    plot_model(
        model, to_file=f"/kaggle/working/images/{name}.png", 
        show_shapes=True, show_layer_names=True, expand_nested=True, dpi=90
    )
    model.compile(optimizer="adamw", loss="mse", metrics=['mape'])
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=config.get("epochs", 0),
        batch_size=config.get("batch_size", 0),
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    model.save(f"./model/{name}.keras")
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(f"model/{name}_loss.csv", encoding='utf-8', index=False)
    # Print the final loss and validation loss
    print(f"Final Training Loss (MSE): {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss (MSE): {history.history['val_loss'][-1]:.4f}")
    # Print the final MAE and validation MAE
    # print(f"Final Training MAE: {history.history['mae'][-1]:.4f}")
    # print(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}")
    
    
    print(f"{name.upper()} predictions:")
    y_pred = np.array(model.predict(y_test))
    assert y_pred.shape == y_test.shape, "something fishy..."
    
    # yeah they should all be normalized
    print("Predictions:")
    print("Normalized:")
    eva_regress(y_test, y_pred)
    
    print("Denomalized:")
    y_test = denormalize(scaler, y_test)
    y_pred = denormalize(scaler, y_pred)
    eva_regress(y_test, y_pred)


import sys 

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "gru"],
        help="Which model architecture to train."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size."
    )
    args = parser.parse_args()

    # select model
    if args.model == "lstm":
        model = get_lstm(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)
    else:
        model = get_gru(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, NUM_LOCATION)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    train_model(
        model,
        X_train_n, y_train_n,
        X_val_n, y_val_n,
        X_test_n, y_test_n,
        args.model,
        config
    )

if __name__ == '__main__':
    main(sys.argv)