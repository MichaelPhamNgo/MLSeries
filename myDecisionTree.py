#################################################################
#   Using decision tree application to predict a test data
#   Training data
#   #,Weight,Height,Blood Pressure,Action,Heart Disease
#   1,light,medium,medium,high,no
#   2,heavy,short,high,low,yes
#   3,light,short,high,low,yes
#   4,heavy,tall,high,medium,no
#   5,light,tall,high,high,no
#   6,medium,short,medium,high,no
#   7,medium,medium,medium,low,no
#   8,heavy,short,low,high,yes
#   Test data
#   9,light,tall,medium,low,?
#   10,light,tall,medium,high,?
#################################################################

#################################################################
#   Step 1: Read training data from DecisionTreeSampleData.csv
#   Step 2: Convert training data to number
#   Step 3: Build model based on training data
#   Step 4: Predict a test data
#################################################################
import pandas as pd
from sklearn import tree

# Step 1: Read training data from DecisionTreeSampleData.csv
df = pd.read_csv('DecisionTreeSampleData.csv')

# Step 2: - Map weight training data to number
#         - Map height training data to number
#         - Map blood pressure training data to number
#         - Map action training data to number
#         - Map heart disease training data to number
map_weight_to_int = {'light':10,'medium':11,'heavy':12}
map_height_to_int = {'short':13,'medium':14,'tall':15}
map_pressure_to_int = {'low':16,'medium':17,'high':18}
map_action_to_int = {'low':19,'medium':20,'high':21}
map_disease_to_int = {'no':0,'yes':1}

df["weightInt"] = df["Weight"].replace(map_weight_to_int)
df["heightInt"] = df["Height"].replace(map_height_to_int)
df["pressureInt"] = df["Blood Pressure"].replace(map_pressure_to_int)
df["actionInt"] = df["Action"].replace(map_action_to_int)
df["diseaseInt"] = df["Heart Disease"].replace(map_disease_to_int)

#   The header # , Weight, Height, Blood Pressure, Action, Heart Disease, weightInt, heightInt, pressureInt, actionInt, diseaseInt
#              0    1       2           3           4           5           6           7           8           9           10
#   Step 3: Build model based on training data
features = list(df.columns[6:10])
X = df[features]
y = df["diseaseInt"]

# Train a decision tree and compute its training accuracy
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

#   Step 4: Predict a test data in row 9
predictData = pd.DataFrame({'Weight':['light'],
                            'Height':['tall'],
                            'Blood Pressure':['medium'],
                            'Action':['low']})

predictData["weightInt"] = predictData["Weight"].replace(map_weight_to_int)
predictData["heightInt"] = predictData["Height"].replace(map_height_to_int)
predictData["pressureInt"] = predictData["Blood Pressure"].replace(map_pressure_to_int)
predictData["actionInt"] = predictData["Action"].replace(map_action_to_int)

#   The header Weight, Height, Blood Pressure, Action, weightInt, heightInt, pressureInt, actionInt
#              0        1           2           3           4           5           6           7
featuresPrediction = list(predictData.columns[4:8])
X_Prediction = predictData[featuresPrediction]

# Predict the heart disease from a test data in row 9
result = clf.predict(X_Prediction)
convertData = pd.DataFrame({0:['no'], 1:['yes']})
print(convertData[result])

# ========================================================================================== #

# Predict a test data in row 10
predictData = pd.DataFrame({'Weight':['light'],
                            'Height':['tall'],
                            'Blood Pressure':['medium'],
                            'Action':['high']})

predictData["weightInt"] = predictData["Weight"].replace(map_weight_to_int)
predictData["heightInt"] = predictData["Height"].replace(map_height_to_int)
predictData["pressureInt"] = predictData["Blood Pressure"].replace(map_pressure_to_int)
predictData["actionInt"] = predictData["Action"].replace(map_action_to_int)

#   The header Weight, Height, Blood Pressure, Action, weightInt, heightInt, pressureInt, actionInt
#              0        1           2           3           4           5           6           7
featuresPrediction = list(predictData.columns[4:8])
X_Prediction = predictData[featuresPrediction]

# Predict the heart disease from a test data in row 9
result = clf.predict(X_Prediction)
convertData = pd.DataFrame({0:['no'], 1:['yes']})
print(convertData[result])