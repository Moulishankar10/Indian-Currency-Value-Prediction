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