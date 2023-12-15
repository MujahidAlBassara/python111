# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:21:10 2023

@author: Zainon
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv("economic_data.csv")
print(data)
print(data.describe())
plt.scatter(data['Year'],data['GDP'] )

from  sklearn.linear_model import LinearRegression
model1=LinearRegression()
x=np.array(data['Year']).reshape(-1,1)
y=np.array(data['GDP'] )
model1.fit(x,y)
#m
print(model1.coef_)
#b
print(model1.intercept_)
plt.scatter(x, y)
plt.plot(x,model1.predict(x),linestyle='dotted')

model1.predict([[2030]])
model1.predict([[2002]])

