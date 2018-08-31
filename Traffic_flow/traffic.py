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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
import argparse

FILE_PATH = 'traffic_flow_data.csv'

def traffic_regression():
	traffic_information = pd.read_csv(FILE_PATH)

	t_features = traffic_information.iloc[:, np.arange(450)].copy()
	t_result = traffic_information.iloc[:,450].copy()

	# Obtained from tutorial code
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

	traffic = pd.concat([feature_prep, t_result.to_frame()], axis=1)

	y = traffic['Segment23_(t+1)']

	ITERATIONS = 10
	all_mse_ridge = []
	all_mse_ord = []
	all_var_ridge = []
	all_var_ord = []

	K_VAL = 400

	for i in range(ITERATIONS):
		x = SelectKBest(f_regression, k=K_VAL).fit_transform(traffic.iloc[:,:450], y)

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

		#=========================================================================
		## Ridge Regression
		#=========================================================================
		reg = Ridge(alpha = .5)

		reg.fit(x_train, y_train)

		y_pred = reg.predict(x_test)
		y_train_pred = reg.predict(x_train)

		all_mse_ridge.append(mean_squared_error(y_test, y_pred))
		all_var_ridge.append(r2_score(y_test, y_pred))


		#=========================================================================
		## Ordinary Regression
		#=========================================================================
		reg = LinearRegression()
		reg.fit(x_train, y_train)

		y_pred = reg.predict(x_test)
		y_train_pred = reg.predict(x_train)
		all_mse_ord.append(mean_squared_error(y_test, y_pred))
		all_var_ord.append(r2_score(y_test, y_pred))

	print('Number of iterations: ', ITERATIONS)
	print('--------------Ridge---------------')
	print("Mean squared error for testing data: %.2f" % (sum(all_mse_ridge)/ITERATIONS))
	print('Variance score for testing data: %.2f' % (sum(all_var_ridge)/ITERATIONS))

	print('--------------Ordinary---------------')
	print("Mean squared error for testing data: %.2f" % (sum(all_mse_ord)/ITERATIONS))
	print('Variance score for testing data: %.2f' % (sum(all_var_ord)/ITERATIONS))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Input relative file name', required=True)

    args = parser.parse_args()

    FILE_PATH = args.input
    traffic_regression()
