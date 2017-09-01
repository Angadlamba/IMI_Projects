import sklearn.tree
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
import pandas as pd
import numpy as np
from time import time
import subprocess

bbc = pd.read_csv('supervisedlearningdataset_13082016.csv', parse_dates=True)

# converting NaN values in the dataset to 0.
for column in bbc.columns:
    bbc[column] = bbc[column].fillna(0)

# removing document id from dataset
bbc = bbc.drop("Document id", 1)

# target variable
target = "label"

# seperating data into features and label
y = bbc.pop(target)
X = bbc
# splitting data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

decision_tree_model = sklearn.tree.DecisionTreeClassifier()

# Cross-validation without Best Parameters:
scores = cross_val_score(decision_tree_model, X_train, y_train, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()))

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """

    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """

    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params

# Grid-Search:
# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 10, 20, 30, 50, 100],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

params = run_gridsearch(X, y, decision_tree_model, param_grid, cv=10)

print("\n-- Best Parameters:")
for k, v in params.items():
    print("parameter: {:<20s} setting: {}".format(k, v))

# Cross-validation with Best Parameters:
print("\n\n-- Testing best parameters [Grid]...")
decision_tree_model = sklearn.tree.DecisionTreeClassifier(**params)
scores = cross_val_score(decision_tree_model, X_train, y_train, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()))

# Visualize Decision Tree
def visualize_tree(tree, feature_names, fn="dt"):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn Decision Tree.
    feature_names -- list of feature names.
    fn -- [string], root of filename, default `dt`.
    """
    dotfile = fn + ".dot"
    pngfile = fn + ".png"

    with open(dotfile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dotfile, "-o", pngfile]

    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, "
             "to produce visualization")

visualize_tree(decision_tree_model, bbc.columns, fn="decision_tree")
