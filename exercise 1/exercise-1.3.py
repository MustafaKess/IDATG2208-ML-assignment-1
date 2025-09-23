# Exercise 1.3 - Linear Regression 
# Q1.3.1 Fit a simple linear regression model using gradiant descent to predict the quality of wine based on chlorides feature
import pandas as pd
import numpy as np

df = pd.read_csv("archive/WineQT.csv")
X = df[['chlorides']].values
y = df['quality'].values
X_b = np.c_[np.ones((X.shape[0], 1)), X]  

# Gradient Descent
learning_rate = 0.01
n_iterations = 1000
m = len(y)
theta = np.random.randn(2)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print("Learned parameters:", theta)