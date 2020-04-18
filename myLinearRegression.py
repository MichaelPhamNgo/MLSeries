#####################################################
#   Mortgage interest rates and home price          #
#                                                   #
#   Year    Interest rate (%)   Median home price   #
#   1988    10.30               $183,800            #
#   1989    10.30               $183,200            #
#   1990    10.10               $174,900            #
#   1991    9.30                $173,500            #
#   1992    8.40                $172,900            #
#   1993    7.30                $173,200            #
#   1994    8.40                $173,200            #
#   1995    7.90                $169,700            #
#   1996    7.60                $174,500            #
#   1997    7.60                $177,900            #
#   1998    6.90                $188,100            #
#   1999    7.40                $203,200            #
#   2000    8.10                $230,200            #
#   2001    7.00                $258,200            #
#   2002    6.50                $309,800            #
#   2003    5.80                $329,800            #
#   Average 8.05625             $204,706.25         #
#####################################################

#####################################################
# Median home price = Weight * Interest rate + Bias #
#  Linear Regression:  y = mx + b                   #
#  Cost Function MSE = (1/n) Sum (yi - (mxi + b))^2 #
#  Find Gradient descent                            #
#   MSE'(m,b) = [df/dm, df/db]                      #
#             = [(1/n) Sum(-2xi(yi - (mxi+b))),     #
#                   (1/n) Sum(-2(yi - (mxi+b))]     #
#####################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read data in the file myLinearRegression.csv
df = pd.read_csv('myLinearRegression.csv')

# read column Interest rate data
X = df.values[:, 1]

# read column Median home price data
Y = df.values[:, 2]

# show data in a graph
plt.scatter(X, Y, marker='o')
plt.show()

# convert X data to a 2D array
X = np.array(X).reshape((-1,1))

# convert X data to a 1D array
Y = np.array(Y)

# using Linear Regression in sklearn library
model = LinearRegression()
model.fit(X, Y)

# Slope of Linear Regression
print('Slope:', model.coef_)

# Interception of Linear Regression
print('Intercept:', model.intercept_)

# R^2 of Linear Regression
print('R^2:', model.score(X, Y))

# Check myLinearRegression.xlsx to compare result

# Predict Median home price with Interest rate (%) = 8.5
print('predicted response:', model.predict([[8.5]]))


