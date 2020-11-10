# INDIAN CURRENCY VALUE PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv(data/data.csv)

print("\nEnter the following details as what you want to predict!")
input_month = input("\nEnter the time period (MM-YYYY) : ")

x = []
initial_str = data["Month"][0]
initial = dt.datetime(int(initial_str[-4:]),int(initial_str[:2]),1)

x_str = dt.datetime(int(input_month[-4:]),int(input_month[:2]),1)
x_pred = (x_str.year - initial.year) * 12 + (x_str.month - initial.month)

for i in range(len(data["Month"])):
    final_str = data["Month"][i]
    final = dt.datetime(int(final_str[-4:]),int(final_str[:2]),1)
    diff = (final.year - initial.year) * 12 + (final.month - initial.month)
    x.append(diff)

x = np.array(x,dtype=int)
x = x.reshape(len(x),1)

y = data["Price"].values
y = np.array(y,dtype=int)
y = y.reshape(len(y),1)