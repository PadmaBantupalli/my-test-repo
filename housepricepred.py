import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price_dataset = fetch_california_housing()

#print(house_price_dataset)
houseprice_df = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
houseprice_df[house_price_dataset.target_names] = pd.DataFrame(house_price_dataset.target, columns=house_price_dataset.target_names)
print(houseprice_df[house_price_dataset.target_names])
'''
print(houseprice_df)
print(houseprice_df.shape)
print(houseprice_df.isnull().sum())
print(houseprice_df.describe())'
'''
#houseprice_df.head()
houseprice_data_corr = houseprice_df.corr()
#print(houseprice_data_corr)
#plt.figure(figsize=(10,10))
#sns.heatmap(houseprice_data_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
#plt.show()
x = houseprice_df.drop(['MedHouseVal'], axis=1)
y = houseprice_df['MedHouseVal']
#print(x)
#print(y)
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2, random_state=2) 
print(x.shape, X_train.shape, X_test.shape)
model = XGBRegressor()
model.fit(X_train, Y_train)
train_data_prediction = model.predict(X_train)
score_1 = metrics.r2_score(Y_train, train_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, train_data_prediction)
print("R Squared error for train data", score_1)
print("mean absolute error for train data", score_2)
test_data_prediction = model.predict(X_test)
score_1_test = metrics.r2_score(Y_test, test_data_prediction)
score_2_test = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R Squared error for test data", score_1_test)
print("mean absolute error for test data", score_2_test)

plt.scatter(Y_train, train_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()