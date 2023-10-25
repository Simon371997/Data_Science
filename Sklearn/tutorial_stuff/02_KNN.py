import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('./tutorial_data/CAR_EVALUATION.csv')
print(data.head())

X = data[['buying', 'maint', 'safety']].values
y = data[['Target']]

# Converting data
## Method 1  (LabelEncoder)
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

## Method 2 (Mapping)
label_maping = {
    'unacc':0, 'acc':1, 'good':2, 'vgood':3
}
y['Target'] = y['Target'].map(label_maping)
y = np.array(y)
print(y)

