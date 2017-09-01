import sklearn.tree
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import pandas as pd
from time import time


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

for max_depth in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]:
    decision_tree_model = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
    t0 = time()
    decision_tree_model.fit(X_train, y_train)
    print "decision_tree_" + str(max_depth) + ":" + str(time() - t0)

    t0 = time()
    scores = cross_val_score(decision_tree_model, X_train, y_train, cv=10)
    print "cross_validation_" + str(max_depth) + ":" + str(time() - t0)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                              scores.std()))

    # visualize_tree(decision_tree_model, bbc.columns, fn="decision_tree_" + str(max_depth))
