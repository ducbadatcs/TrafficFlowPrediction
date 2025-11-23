"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
# from keras.models import load_model
from keras import Sequential
from keras.saving import load_model
from keras.utils import plot_model
import sklearn.metrics as metrics

from matplotlib.dates import AutoDateFormatter
import matplotlib.pyplot as plt
from typing import Any
warnings.filterwarnings("ignore")


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y = [x for x in y_true if x > 0]
    y_pred = np.array([y_pred[i] for i in range(len(y_true)) if y_true[i] > 0])

    num: int = len(y_pred)
    sums: float = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true: np.ndarray, y_pred: np.ndarray):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true: np.ndarray, y_preds: np.ndarray, names: list[Any]):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = AutoDateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def main():
    if os.path.exists("scats.csv")


if __name__ == '__main__':
    main()
