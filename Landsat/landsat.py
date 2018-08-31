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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, classification_report
import argparse

TEST_PERCENTAGE = 0.2
MAX_PERCENTAGE = 100
FILE_PATH = 'lantsat.csv'

PERCEPTRON = "PERCEPTRON"
NAIVE = "NAIVE"
NEAR_NEIGHBOUR = "NEAR_NEIGHBOUR"
SVM = "SVM"
TREES = "DECISION TREES"

def setup():
    sat_information = pd.read_csv(FILE_PATH, header=-1)

    s_features = sat_information.iloc[:, np.arange(36)].copy()
    s_result = sat_information.iloc[:,36].copy()

    # Obtained from tutorial code
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names].values


    full_pipe = Pipeline([
            ('selector', DataFrameSelector(list(s_features))),
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler()),
        ])

    s_result.reset_index(drop=True, inplace=True)

    feature_prep = pd.DataFrame(data=full_pipe.fit_transform(s_features), index=np.arange(1,6001))
    feature_prep.reset_index(drop=True, inplace=True)

    sat = pd.concat([feature_prep, s_result.to_frame()], axis=1)
    return sat

def landsat_classification(type):
    sat = setup()

    ITERATIONS = 20

    x = sat.iloc[:,:35]
    y = sat[36]

    all_acc = []

    for i in range(ITERATIONS):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_PERCENTAGE)


        if type == TREES:
            params = {  
                        'max_leaf_nodes':list(range(2,50)),
                        'min_samples_split': [3,4,5],
                    }

            grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params)

        elif type == PERCEPTRON:
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

        elif type == SVM:
            params = {
                'C': [10**x for x in range(-1,1)],
                # 'C': [1],
                'gamma': [10**x for x in range(-1,1)],
                # 'gamma': [1],
            }

            grid_search_cv = GridSearchCV(SVC(), params)
        grid_search_cv.fit(x_train, y_train)

        y_pred = grid_search_cv.predict(x_test)
        all_acc.append(accuracy_score(y_test, y_pred))

    print(type)
    print(classification_report(y_test, y_pred))
    print("The average accuracy over {} iterations is : {:.2f}%.".format(ITERATIONS, 100*(sum(all_acc)/ITERATIONS)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Input relative file name', required=True)
    parser.add_argument('--model', help='enter ["SVM", "DT", "KNN", "PERC", "NAIVE"]', required=True)

    args = parser.parse_args()

    FILE_PATH = args.input

    if args.model == "SVM":
        landsat_classification(SVM)
    elif args.model == "DT":
        landsat_classification(TREES)
    elif args.model == "KNN":
        landsat_classification(NEAR_NEIGHBOUR)
    elif args.model == "PERC":
        landsat_classification(PERCEPTRON)
    elif args.model == "NAIVE":
        landsat_classification(NAIVE)
    else:
        print('Please enter ["SVM", "DT", "KNN", "PERC", "NAIVE"] for model')