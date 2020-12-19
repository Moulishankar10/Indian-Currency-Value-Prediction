# INDIAN CURRENCY VALUE PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt

# IMPORTING DATA
data = pd.read_csv("data/data.csv")

# PREPROCESSING DATA
x = []
initial_str = data["Month"][0]
initial = dt.datetime(int(initial_str[-4:]),int(initial_str[:2]),1)
 
for i in range(len(data["Month"])):
    final_str = data["Month"][i]
    final = dt.datetime(int(final_str[-4:]),int(final_str[:2]),1)
    diff = (final.year - initial.year) * 12 + (final.month - initial.month)
    x.append(diff)

y = data["Price"].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# RESHAPING THE DATA
x_train = np.reshape(x_train, (-1,1))
x_val = np.reshape(x_val, (-1,1))
y_train = np.reshape(y_train, (-1,1))
y_val = np.reshape(y_val, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

xtrain_scaled = scaler_x.fit_transform(x_train)
ytrain_scaled = scaler_y.fit_transform(y_train)
xval_scaled = scaler_x.fit_transform(x_val)
