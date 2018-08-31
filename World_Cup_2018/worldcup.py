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


worldcup_information = pd.read_csv('2018 worldcup.csv')
worldcup_information.drop(['Date', 'Location', 'Phase', 'Team1', 'Team1_Continent', 'Team2', 'Team2_Continent', 'Normal_Time'], axis=1, inplace=True)
# worldcup_information = occ
print(worldcup_information.describe())

print(worldcup_information.shape)
# print(worldcup_information)


wc_features = worldcup_information.iloc[:, np.arange(21)].copy()
wc_goals = worldcup_information.iloc[:,21].copy()
wc_result = worldcup_information.iloc[:,22].copy()
# print(wc_result)
# print(wc_goals)

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


full_pipe = Pipeline([
		('selector', DataFrameSelector(list(wc_features))),
		('imputer', Imputer(strategy='median')),
		('std_scaler', StandardScaler()),
	])

wc_result.reset_index(drop=True, inplace=True)

feature_prep = pd.DataFrame(data=full_pipe.fit_transform(wc_features), index=np.arange(1,65))
feature_prep.reset_index(drop=True, inplace=True)
# print(feature_prep)
print('a')
print(feature_prep.shape)
print(wc_result.shape)

worldcup = pd.concat([feature_prep, wc_goals, wc_result.to_frame()], axis=1)
print(worldcup.shape)


# print(worldcup[:10][:])

x = worldcup.iloc[:,:20]
y = worldcup['Match_result']

# print(x[:10][:])
# print("x", x.shape)

# print(y[:10][:])
# print("y", y.shape)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#=========================================================================
## Decision Tree Classification
#=========================================================================
params = {	
			'max_leaf_nodes':list(range(2,3)),
			'min_samples_split': [5],
		}

# grid_searchgrid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params)
grid_search_cv.fit(x_train, y_train)
print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(x_test)
print("The prediction accuracy using the decision tree is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))

#=========================================================================
## Perceptron
#=========================================================================
from sklearn.linear_model import Perceptron

params = {
	'alpha':[10**x for x in range(-10,1)],
	'tol': [None],
	'max_iter': [x for x in range(1,5)],
}

grid_search_cv = GridSearchCV(Perceptron(), params)
grid_search_cv.fit(x_train, y_train)
print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(x_test)
print("The prediction accuracy using the Perceptron is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))

#=========================================================================
## Naive Baysian
#=========================================================================
from sklearn.naive_bayes import GaussianNB

params = {}

grid_search_cv = GridSearchCV(GaussianNB(), params)
grid_search_cv.fit(x_train, y_train)
print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(x_test)
print("The prediction accuracy using the Naive Baysean is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))

#=========================================================================
## Nearest Neighbour
#=========================================================================
from sklearn.neighbors import KNeighborsClassifier

params = {
	'n_neighbors': [x for x in range(2,10)],
	'metric': ['minkowski','euclidean','manhattan'],
}

grid_search_cv = GridSearchCV(KNeighborsClassifier(), params)
grid_search_cv.fit(x_train, y_train)
print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(x_test)
print("The prediction accuracy using the Nearest Neighbour is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))

#=========================================================================
## SVM
#=========================================================================
from sklearn.svm import SVC

params = {
	'C': [10**x for x in range(-1,3)],
	# 'C': [1],
	'gamma': [10**x for x in range(-1,2)],
	# 'gamma': [1],
}

grid_search_cv = GridSearchCV(SVC(), params)
grid_search_cv.fit(x_train, y_train)
print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(x_test)
print("The prediction accuracy using the SVM is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))

#=========================================================================
##  Regression
print("REGRESSION")
#=========================================================================
from sklearn.metrics import mean_squared_error, r2_score

worldcup.drop(['Match_result'], axis=1, inplace=True)


x = worldcup.iloc[:,:20]
y = worldcup['Total_Scores']

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
