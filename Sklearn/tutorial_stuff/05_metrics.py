import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./tutorial_data/creditcard.csv')[:80000]
print(df.head(3))

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values


print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')
print(f'Fraud Cases: {y.sum()}')
# unbalanced Dataset (just 196 Fraud Cases in 80.000 entries) (0 = non fraud, 1 = fraud)


from sklearn.linear_model import LogisticRegression
mod = LogisticRegression(class_weight ={0:1, 1:2} ,max_iter=1000) #
mod.fit(X, y).predict(X).sum()

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0:1, 1:v} for v in range(1,4)]},
    cv = 4,
    n_jobs=-1
)
grid.fit(X, y)