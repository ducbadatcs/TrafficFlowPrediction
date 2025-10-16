"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import keras
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from typing import Any, List
warnings.filterwarnings("ignore")


def train_model(model: keras.Sequential, X_train: np.ndarray, y_train: np.ndarray, name: str, config: dict[str, Any]):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(f"model/{name}.keras")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"model/{name}_loss.csv", encoding='utf-8', index=False)
    
    # read loss
    
    print(f"Metrics: {df.columns}")
    # print(df.columns)
    
    
    


def train_seas(models: List[keras.Sequential], X_train: np.ndarray, y_train: np.ndarray, name:str , config: dict[str, Any]):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(inputs=p.inputs,
                                       outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag: int = 12
    config: dict[str, Any] = {"batch": 256, "epochs": 600}
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)
    print(X_train)
    print(y_train)
    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([lag, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)
    elif args.model == "cnn":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_cnn([lag, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    


if __name__ == '__main__':
    main(sys.argv)
