# INDIAN CURRENCY VALUE PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R

# IMPORTING MODULES
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# PREDICTING THE MODEL
y_est = model.predict(x_scaled)
y_est = scaler_y.inverse_transform(y_est)

# VISUALISING THE MODEL PERFORMANCE
plt.plot(x,y, color = 'blue')
plt.plot(x,y_est, color = 'red')
plt.title('Indian Currency Value Prediction')
plt.xlabel('Time')
plt.ylabel('GMSL')
plt.legend(['Actual Data', 'Predicted Data'], loc='upper left')
plt.show()