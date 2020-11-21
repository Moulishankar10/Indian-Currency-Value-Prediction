# INDIAN CURRENCY VALUE PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R
# VIGNESHWAR RAVICHANDAR

# IMPORTING MODULES
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def mlmodel():

    # IMPORTING DATA
    data = pd.read_csv("data/data.csv")

    # INPUT DATA
    print("\nEnter the following details as what you want to predict!")
    input_month = input("\nEnter the time period (MM-YYYY) : ")

    # PREPROCESSING DATA
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

    # FITTING MODEL
    regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
    regressor.fit(x, y)

    # PREDICTING MODEL
    res = regressor.predict([[x_pred]])

    # DISPLAYING RESULTS
    print(f"The Predicted INR per USD on {input_month} --- {float(res)}")

    # TO VISUALISE THE ACCURACY

    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~POLYNOMIAL REGRESSION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.plot(data["Month"], data["Price"], color = 'red')
    plt.plot(data["Month"], mod.predict(model.fit_transform(x)), color = 'green') 
    plt.scatter(x_pred, res, color = 'blue')
    plt.show()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## ACCURACY SCORE PREDICTION
    print("Accuracy : ",regressor.score(x,y))

    
    ## GRAPHICAL VISUALISATION
    x_grid = np.arange(min(x), max(x), 0.01)
    x_grid = x_grid.reshape((len(x_grid), 1))
    plt.plot(x, y, color = 'red')
    plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
    plt.scatter(x_pred, res, color='green')
    plt.show()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
print(
    '''

                  HAPPY TO HAVE YOU ON INDIAN CURRENCY VALUE PREDICTION !

''')
mlmodel()