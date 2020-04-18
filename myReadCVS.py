import numpy as np
import pandas as pd

# read data in the file UNSW_NB15_testing-set.csv
df = pd.read_csv('UNSW_NB15_testing-set.csv', header=None, low_memory=False)

# read data at column 3
df = df[3]

# read the first 200 rows data
df = pd.read_csv('UNSW_NB15_testing-set.csv', skiprows=0, nrows=200, low_memory=False)

# write the data to test.csv
df.to_csv('test.csv')