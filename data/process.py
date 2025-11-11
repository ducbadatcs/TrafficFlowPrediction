# 1. Read data
import pandas as pd

# process the scats data
def process_scats_data(where: str = "./Scats Data October 2006.xls") -> pd.DataFrame:
    sheets = pd.read_excel(where, sheet_name=None)
    removed_days = sheets["Notes"]["Unnamed: 1"].dropna().to_list()[5:]
    removed_days = [int(i) for i in removed_days]
    # removed_days
    df_info = sheets["Summary Of Data"].loc[2:]
    df_info.columns = df_info.iloc[0]
    df_info = df_info[1:]
    df_info["SCATS Number"] = df_info["SCATS Number"].ffill()
    
    df_info.dropna(axis=1, inplace=True)
    df_info = df_info.convert_dtypes().infer_objects()
    df_info_filtered = df_info[df_info["Total"].between(7, 31)]
    
    df = sheets["Data"]
    df.columns = pd.Series(df.loc[0])
    df = df.loc[1:]
    
    df.loc[:, 'Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.convert_dtypes()
    df = df.loc[df["Location"].isin(df_info_filtered["Location"])]
    df.loc[:, "Identifier"] = df["SCATS Number"].astype(str) + " - " + df["HF VicRoads Internal"].astype(str)
    df.columns = df.columns.str.strip()
    return df


print(process_scats_data())