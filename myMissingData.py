import numpy as np
import pandas as pd
# library fill missing data
from sklearn.impute import SimpleImputer

# read data from myMissingData.csv
data = pd.read_csv('myMissingData.csv', header=None)

# convert data to an 1d array
X = data.values

# fill missing data by compute the average. strategy = 'mean' -> 'most_frequent'
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
result = imp.transform(X)
print(result)

