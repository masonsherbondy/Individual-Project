import numpy as np
import pandas as pd
pd.options.display.max_columns = 61
pd.options.display.max_rows = 61
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import mason_functions as mf
import explore
import wrangle
import scale 

from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




def tree_train_validate(X_train, y_train, X_validate, y_validate, tree_model, min_samples_leaf):

    depth_range = range(1, 11)    # set a range of depths to explore

    scores = []    # set an empty list for validate scores

    metrics = []    # set an empty list for dictionaries

    for depth in depth_range:    # commence loop through different max depths for Decision Tree

        model = tree_model(max_depth = depth, min_samples_leaf = min_samples_leaf, random_state = 421) # create object

        model.fit(X_train, y_train)    # fit object

        scores.append(model.score(X_validate, y_validate))    # add validate scores to scores list

        in_sample_accuracy = model.score(X_train, y_train)    # calculate accuracy on train set

        out_of_sample_accuracy = model.score(X_validate, y_validate)    # calculate accuracy on validate set

        output = {                                       # create dictionary with max_depth,
            'max_depth': depth,                          # train set accuracy, and validate accuracy
            'train_accuracy': in_sample_accuracy,        
            'validate_accuracy': out_of_sample_accuracy
        }

        metrics.append(output)    # add dictionaries to list

    plt.figure()    # create figure

    plt.xlabel('depth')    # label x-axis

    plt.ylabel('accuracy')    # label y-axis

    plt.scatter(depth_range, scores, color = 'indianred') # plot relatiosnhip between depth range and validate accuracy

    plt.xticks([0, 2, 4, 6, 8, 10])    # customize x-axis label ticks

    plt.title('Validate Accuracy')    # title

    plt.show();

    metrics_df = pd.DataFrame(metrics)    # form dataframe from scores data

    metrics_df = metrics_df.set_index('max_depth')

    metrics_df['difference'] = metrics_df.train_accuracy - metrics_df.validate_accuracy   # create column of values
                                                                        # for difference between train and validate
    print(metrics_df)         # view metrics dataframe


def KNN_metrics(X_train, y_train, X_validate, y_validate, weights):
    k_range = range(1, 21)
    scores = []
    metrics = []
    for k in k_range:
        titan_knn = KNeighborsClassifier(n_neighbors = k, weights = weights)
        titan_knn.fit(X_train, y_train)

        scores.append(titan_knn.score(X_validate, y_validate))

        in_sample_accuracy = titan_knn.score(X_train, y_train)
        out_of_sample_accuracy = titan_knn.score(X_validate, y_validate)
        output = {
            'n_neighbors': k,
            'train_accuracy': in_sample_accuracy,
            'validate_accuracy': out_of_sample_accuracy
        }

        metrics.append(output)

    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.title('Validate Accuracy')
    plt.show();
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index('n_neighbors')
    metrics_df['difference'] = metrics_df.train_accuracy - metrics_df.validate_accuracy
    print(metrics_df)