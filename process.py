import pandas as pd
import numpy as np 
from typing import List
from sklearn.preprocessing import StandardScaler

def process() -> pd.DataFrame:
    sheets = pd.read_excel("./data/Scats Data October 2006.xls", sheet_name=None)
    removed_days = sheets["Notes"]["Unnamed: 1"].dropna().to_list()[5:]
    removed_days = [int(i) for i in removed_days]
    
    df_info = sheets["Summary Of Data"].loc[2:]
    df_info.columns = df_info.iloc[0]
    df_info = df_info[1:]
    df_info["SCATS Number"] = df_info["SCATS Number"].ffill()
    
    df_info.dropna(axis=1, inplace=True)
    df_info = df_info.convert_dtypes().infer_objects()
    
    df_info_filtered = df_info[df_info["Total"].between(7, 31)]
    df = sheets["Data"]
    df.columns = pd.Series(df.loc[0])
    
    # I love formatting
    df = df.loc[1:]
    df.loc[:, 'Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.convert_dtypes()
    
    # filter good locations
    df = df.loc[df["Location"].isin(df_info_filtered["Location"])]
    
    # add location identifier: SCARS Number + VicRoads Internal
    df.loc[:, "Identifier"] = df["SCATS Number"].astype(str) + " - " + df["HF VicRoads Internal"].astype(str)
    df.columns = df.columns.str.strip()
    df = df.fillna(0)
    df.reset_index(drop=True)
    df.to_csv("./data/scats.csv")
    return df
    
def make_windows(df: pd.DataFrame) -> np.ndarray:
    vcols = [f"V{str(i).zfill(2)}" for i in range(96)]
    DAY_LENGTH = 96 # intervals per day
    MONTH_LENGTH = 31 # days in oct
    d = {}
    for id, group in df.groupby(by="Identifier"):
        # identifier + flow
        flow = list(group[vcols].fillna(0).to_numpy().flatten())
        while len(flow) < DAY_LENGTH * MONTH_LENGTH: flow.append(0)
        d[id] = flow
    d = {i: np.array(d[i]) for i in d}
    flow = pd.DataFrame(d)
    data = flow.to_numpy().astype(float)
    windows = []
    WEEK_LENGTH = DAY_LENGTH * 7  # 1 week
    WINDOW_LENGTH = WEEK_LENGTH
    for i in range(WINDOW_LENGTH, len(data) + 1):
        windows.append(data[i - WINDOW_LENGTH : i])
    windows = np.array(windows)
    NUM_WINDOW, _, NUM_LOCATION = windows.shape
    for name, val in zip(["Number of windows", "Window Length", "Number of locations"], windows.shape):
        print(f"{name}: {val}")
    return windows

def make_dataset(windows: np.ndarray) -> List[np.ndarray]:
    NUM_WINDOW, WINDOW_LENGTH, NUM_LOCATION = windows.shape
    HORIZON = 24  # assuming hourly data
    # Input is all but last HORIZON steps
    X = np.array(windows[:, :-HORIZON, :])  # shape: (num_windows, WINDOW_LENGTH - HORIZON, NUM_LOCATIONS)
    
    # Output is the next HORIZON steps
    y = np.array(windows[:, -HORIZON:, :])  # shape: (num_windows, HORIZON, NUM_LOCATIONS)
    
    train_size = int(NUM_WINDOW * 0.7)
    val_size = int(NUM_WINDOW * 0.15)
    X_train = X[:train_size]
    X_val   = X[train_size : train_size + val_size]
    X_test  = X[train_size + val_size :]
    y_train = y[:train_size]
    y_val   = y[train_size : train_size + val_size]
    y_test  = y[train_size + val_size :]
    print("Shapes: ")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape) 
    return [X_train, X_val, X_test, y_train, y_val, y_test]


windows = make_windows(process())


if __name__ == "__main__":
    process()