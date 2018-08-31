import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


traffic_information = pd.read_csv('traffic_flow_data.csv')
print(traffic_information.describe())

t_features = traffic_information.iloc[:, np.arange(450)].copy()
# print(t_features)
t_result = traffic_information.iloc[:,450].copy()
# print(t_result)

# exit()
# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


pipe1 = Pipeline([
		('selector', DataFrameSelector(list(t_features))),
		('imputer', Imputer(strategy='median')),
		('std_scaler', StandardScaler()),
	])

t_result.reset_index(drop=True, inplace=True)
feature_prep = pd.DataFrame(data=pipe1.fit_transform(t_features), index=np.arange(1,7501))
feature_prep.reset_index(drop=True, inplace=True)

print(feature_prep.shape)
print(t_result.to_frame().shape)

traffic = pd.concat([feature_prep, t_result.to_frame()], axis=1)


# print(traffic[:][:])
# print(traffic.shape)

x = traffic.iloc[:,:450]
y = traffic['Segment23_(t+1)']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#=========================================================================
## Ridge Regression
#=========================================================================
from sklearn.linear_model import Ridge

reg = Ridge(alpha = .5)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
y_train_pred = reg.predict(x_train)
print('--------------Ridge---------------')
print('Coefficients and Intercept are: ', reg.coef_,"   ",reg.intercept_,' respectively')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score for testing data: %.2f' % r2_score(y_test, y_pred))
print("Mean squared error for training data: %.2f"
      % mean_squared_error(y_train, y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' % r2_score(y_train, y_train_pred))

#=========================================================================
## Ordinary Regression
#=========================================================================
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
y_train_pred = reg.predict(x_train)

print('--------------Ordinary---------------')
print('Coefficients and Intercept are: ', reg.coef_,"   ",reg.intercept_,' respectively')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score for testing data: %.2f' % r2_score(y_test, y_pred))
print("Mean squared error for training data: %.2f"
      % mean_squared_error(y_train, y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' % r2_score(y_train, y_train_pred))
