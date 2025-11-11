import numpy as np
import pandas as pd
from data.process import process_scats_data

df = process_scats_data("./data/Scats Data October 2006.xls")
print(df)