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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

TEST_PERCENTAGE = 0.2
MAX_PERCENTAGE = 100
FILE_PATH = '2018 worldcup.csv'

PERCEPTRON = "PERCEPTRON"
NAIVE = "NAIVE"
NEAR_NEIGHBOUR = "NEAR_NEIGHBOUR"
SVM = "SVM"
TREES = "TREES"


def setup(): #preprocessing
    worldcup_information = pd.read_csv(FILE_PATH)
    worldcup_information.drop(['Date', 'Location', 'Phase', 'Team1', 'Team1_Continent', 'Team2', 'Team2_Continent', 'Normal_Time'], axis=1, inplace=True)

    wc_features = worldcup_information.iloc[:, np.arange(21)].copy()
    wc_goals = worldcup_information.iloc[:,21].copy()
    wc_result = worldcup_information.iloc[:,22].copy()

    # Obtained from tutorial code
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

    worldcup = pd.concat([feature_prep, wc_goals, wc_result.to_frame()], axis=1)

    return worldcup

def world_cup_classification(type):
    worldcup = setup()

    x = worldcup.iloc[:,:20]
    y = worldcup.iloc[:, -1]

    all_acc = []
    ITERATIONS = 20

    for i in range(ITERATIONS):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_PERCENTAGE)

        if type == PERCEPTRON:
            params = {
                'alpha':[10**x for x in range(-10,1)],
                'tol': [None],
                'max_iter': [x for x in range(1,5)],
            }

            grid_search_cv = GridSearchCV(Perceptron(), params)

        elif type == NAIVE:
            params = {}

            grid_search_cv = GridSearchCV(GaussianNB(), params)

        elif type == NEAR_NEIGHBOUR:
            params = {
                'n_neighbors': [x for x in range(2,10)],
                'metric': ['minkowski','euclidean','manhattan'],
            }

            grid_search_cv = GridSearchCV(KNeighborsClassifier(), params)

        elif type == TREES:
            params = {  
                        'max_leaf_nodes':list(range(2,3)),
                        'min_samples_split': [5],
                    }

            grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params)

        elif type == SVM:
            params = {
                'C': [10**x for x in range(-1,3)],
                'gamma': [10**x for x in range(-1,2)],
            }

            grid_search_cv = GridSearchCV(SVC(), params)

        grid_search_cv.fit(x_train, y_train)

        y_pred = grid_search_cv.predict(x_test)
        all_acc.append(accuracy_score(y_test, y_pred))

    print(type)
    print(classification_report(y_test, y_pred))
    print("The average accuracy over {} iterations is : {:.2f}%.".format(ITERATIONS, 100*(sum(all_acc)/ITERATIONS)))


def world_class_regression():
    worldcup = setup()
    worldcup.drop(['Match_result'], axis=1, inplace=True)

    y = worldcup['Total_Scores']
    x = SelectKBest(f_regression, k=19).fit_transform(worldcup.iloc[:,:20], y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #=========================================================================
    ## Ridge Regression
    #=========================================================================
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
    print('Variance score for training data: %.2f' % r2_score(y_train, y_train_pred))

    #=========================================================================
    ## Ordinary Regression
    #=========================================================================
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
    print('Variance score for training data: %.2f' % r2_score(y_train, y_train_pred))


world_cup_classification(PERCEPTRON)
world_cup_classification(NAIVE)
world_cup_classification(NEAR_NEIGHBOUR)
world_cup_classification(TREES)
world_cup_classification(SVM)
world_class_regression()
