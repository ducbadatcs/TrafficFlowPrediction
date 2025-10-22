"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train_csv: str, test_csv: str, lags: int, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(train_csv, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test_csv, encoding='utf-8').fillna(0)
    
    print("values:", np.array(df1[attr].values))
    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(df1[attr].values).reshape(-1, 1))
    flow1 = scaler.transform(np.array(df1[attr].values).reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(np.array(df2[attr].values).reshape(-1, 1)).reshape(1, -1)[0]
    print("value shapes: ", np.array(df1[attr].values).shape)
    print("F1:", np.array(flow1).shape)
    print("F2:", np.array(flow2).shape)

    train_list, test_list = [], []
    for i in range(lags, len(flow1)):
        train_list.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test_list.append(flow2[i - lags: i + 1])
        

    train = np.array(train_list)
    test = np.array(test_list)
    print(f"Train:", train)
    print(f"Test:", test)
    if shuffle: np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = process_data(
        "./train.csv", "./test.csv", 10, False)
    # print(X_train.shape)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
    print(f"X_test = {X_test}")
    print(f"y_test = {y_test}")