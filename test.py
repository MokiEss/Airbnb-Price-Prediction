import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import general_preprocessing as prep


df = pd.read_csv('listing.csv')
df = prep.pre_processing(df)
print("number of listing after preprocessing ", len(df))
print("number of features after preprocessing", len(df.columns))
df.to_csv('listings_after_preprocessing.csv', index=False)

# modeling

y = df['price']
x = df.drop(columns=['price'])
baseline_pred = y.mean()
mae_baseline = abs(y - baseline_pred).mean()


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = XGBRegressor()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)
print(mean_absolute_error(ytest, y_pred), mae_baseline)
