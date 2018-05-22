import collections
import copy
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import re
from DataExtraction import *
from io import StringIO
from sklearn import linear_model, pipeline, feature_extraction, metrics, base
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


class DecisionTree(object):
    # flat_list = [item for sublist in l for item in sublist]

    def __init__(self, path, depth, fold_files_path, number_of_fold_files):

        self.fold_files_path = fold_files_path
        self.number_of_fold_files = number_of_fold_files
        self.training_path = path
        c = loadCorpus(path)
        t = DataExtraction(c[0], c[1])
        t.get_X_Y()

        self.X = t.get_X()
        self.y = t.get_Y()
        self.scores_p_a = list()

        if depth == 1:
            self.model = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=15)

        if depth == 4:
            self.model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, max_leaf_nodes=10)

        if depth == 8:
            self.model = tree.DecisionTreeClassifier(max_depth=8, max_leaf_nodes=10, min_samples_leaf=10)

    def copy(self, x):

        return copy.deepcopy(x)

    def create_train_and_test_folds(self):

        # Main function to calculate tr_a and p_a accross 5 files

        fold_directories = [self.fold_files_path + str(s) for s in range(1, self.number_of_fold_files + 1)]
        X_list = []
        y_list = []

        for fold in fold_directories:
            # get all the DATA and label sets from 5 files and store them

            c = loadCorpus(fold)
            t = DataExtraction(c[0], c[1])
            t.get_X_Y()
            X_list.append(t.get_X())
            y_list.append(t.get_Y())

        # turn those list into np arrays of np arrays
        y = np.asarray(y_list)
        X = np.asarray(X_list)

        k_fold = KFold(n_splits=self.number_of_fold_files)

        scores_tr_a = list()
        self.scores_p_a = list()
        max_score = 0
        for k, (train, test) in enumerate(k_fold.split(X_list, y_list)):
            # create label train and test files
            y_out, y_test = y[train], y[test][0]

            # the dimensions should be reorganized to (560,)
            y_train = np.concatenate(y_out).ravel()

            # split features into train and test groups
            X_test, X_train = X[test][0], X[train]

            # Have to reshape train set from (4,140,260) to (560,260)
            X_train = X_train.reshape((X_train.shape[1] * X_train.shape[0], (X_train.shape[2])))

            # This code is to include the tree that had max performance during cross validation.

            score = self.model.fit(X_train, y_train).score(X_test, y_test)

            # This code is to print the tree that had max performance during cross validation.
            # It also prints its confusion matrix to count number of correct predictions.
            # uncomment to visualize the tree and its confusion matrix.
            """ if (score > max_score):
                max_score = score
                self.visualize_tree(self.model, X_train, y_train, feature_names, "Tree_with_Max_CV-Score")
                              #self.measure_performance(self.model, X_test, y_test)

             """

            # add the p_a accuracy score of our SGD Classifier to calculate the mean

            self.scores_p_a.append(score)

            # score the tr_a accuracy of our SGD Classifier

            scores_tr_a.append(self.model.fit(X_train, y_train).score(X_train, y_train))

        print("Mean Accuracy when tested on fifth fold. (i.e. p_a)")
        print(self.scores_p_a)
        pa = np.asarray(self.scores_p_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (pa.mean(), pa.std() * 2))

        print("Mean Accuracy when tested on the folds 1-4. (i.e. tr_a)")
        print(scores_tr_a)
        tra = np.asarray(scores_tr_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (tra.mean(), tra.std() * 2))

    def tune_dt_parameters(self):

        # function for Decision Tree Classifier hyper parameter tuning and plotting with PyCharm SciView

        # the values we would like to test for a parameter
        parameter_values = [None, 2, 3, 4, 5, 7, 8, 10, 11, 12, 15, 18, 20, 25, 30]

        xi = [i for i in range(0, len(parameter_values))]

        parameter_scores = []

        # our original classifier model should be saved if the user wants to run 5-CV Validation with untuned version.
        original_model = self.copy(self.model)

        for param in parameter_values:
            # Let us plot max_leaf_nodes
            self.model = tree.DecisionTreeClassifier(max_depth=8, max_leaf_nodes=param, min_samples_leaf=10)

            # Perform 5-fold cross validation
            self.create_train_and_test_folds()

            parameter_scores.append(np.asarray(self.scores_p_a).mean())

        self.model = original_model

        plt.plot(xi, parameter_scores, marker='o', linestyle='--', color='r', label='Score')

        plt.xlabel('max_leaf_nodes for max depth-8 DT')
        plt.ylabel('Cross-validated accuracy')
        plt.xticks(xi, parameter_values)
        plt.legend()

        plt.show()

    def measure_performance(self, clf, X, y, show_accuracy=False, show_classification_report=False,
                            show_confusion_matrix=True):

        y_pred = clf.predict(X)

        if show_accuracy:
            print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")

        if show_classification_report:
            print("Classification report")
            print(metrics.classification_report(y, y_pred))

        if show_confusion_matrix:
            print("Confusion matrix")
            print(metrics.confusion_matrix(y, y_pred), "\n")

    def visualize_tree(self, clf, X, y, feature_names, class_names, tree_name):
        clf.fit(X, y)

        dot_data = tree.export_graphviz(clf,
                                        feature_names=feature_names, class_names=class_names,
                                        out_file=None,
                                        filled=True,
                                        rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)

        colors = ('turquoise', 'orange')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                dest.set_fillcolor(colors[i])

        graph.write_png(tree_name + ".png")

    def decision_tree_predict(self, test_path):

        # The prediction is made by DT - 8 on test labels
        X_list = []
        y_list = []
        c1 = loadCorpus(self.training_path)
        t1 = DataExtraction(c1[0], c1[1])
        t1.get_X_Y()

        self.X = t1.get_X()
        self.y = t1.get_Y()
        X_list.append(t1.get_X())
        y_list.append(t1.get_Y())
        y = np.asarray(y_list)
        X = np.asarray(X_list)

        # These variables represent training data and labels
        y_train = np.concatenate(y).ravel()
        X_train = X.reshape((X.shape[1] * X.shape[0], (X.shape[2])))

        c2 = load_test_data(test_path)

        # get the test_name - 250D Vector dictionary
        t2 = DataExtraction(c2[0], c2[1])

        # this represents the 260D Vector format of testnames

        t2.test_names_to_vector()

        oldvector_to_test_name_dict = t2.get_test_name_vector_dict()

        self.model.fit(X_train, y_train)
        test_labels_txt = open('test_labels.txt', 'w')
        for (old_test_vector, name) in oldvector_to_test_name_dict.items():
            self.model.fit(X_train, y_train)

            np_new_test_vector = np.asarray(old_test_vector)

            label = (self.model.predict(np_new_test_vector.reshape(1, -1)))

            if int(label[0]) == 0:

                sign = '-'
            else:
                sign = '+'
            test_labels_txt.write(sign + ' ' + str(name) + "\n")


def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)
    return (report_df)

###### USAGE ########

## STEP 1## Create an DT Object
# First parameter --> path of the train folder
# Second Paremeter --> Depth of the tree. Use 1 for Full Decision Tree.
# Third Paremeter --> Path of the fold files (WITHOUT NUMBER!,example below). They must be all under same file directory.
# Fourth Paremeter --> Number of fold files. Used for k-Fold, Default = 5

DT_1 = DecisionTree("badges.modified.data.train",8, "badges.modified.data.fold", 5)

## STEP 2## Cross Validation
# Call this function to get  p_a and tr_a of 5-fold validation
DT_1.create_train_and_test_folds()

## STEP 3##  Parameter Tuning
DT_1.tune_dt_parameters()

##STEP 4##  Test it on test data
# First argument: path of the test set
DT_1.decision_tree_predict("badges.modified.data.test")
