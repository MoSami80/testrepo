# Multiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import os
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# Load the data from a .csv in the same folder
# path = os.getcwd()  # This command gives the current path
path = 'C:\Data Science Course 2021 - Udemy\Part_5_Advanced_Statistical_Methods_(Machine_Learning)\S34_L212'
data = pd.read_csv(path + '/1.02. Multiple linear regression.csv')

# This method generates descriptive statistics.
data.describe()
data.head()

# Define the dependent (output or target) and the independent (input or feature) variables
y = data['GPA']
x = data[['SAT' , 'Rand 1,2,3']]

# Regression
reg = LinearRegression()
reg.fit(x,y)

# Coefficients
reg.coef_

# Intercept
reg.intercept_

# R-squared
reg.score(x, y)

# Adjusted R-squared
r2 = reg.score(x,y)
n = x.shape[0]  # Number of observations
p = x.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1-r2) * (n-1) / (n-p-1)

# Feature Selection
f_statistics = f_regression(x,y)[0]
p_values = f_regression(x,y)[1]
