from time import time

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataExtraction import *
from operator import itemgetter
from sklearn import linear_model, pipeline, feature_extraction, metrics, base
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score, cross_val_predict, ParameterGrid


class SGD(object):
    # flat_list = [item for sublist in l for item in sublist]

    def __init__(self, path, fold_files_path, number_of_fold_files):

        self.fold_files_path = fold_files_path
        self.number_of_fold_files = number_of_fold_files


        c = loadCorpus(path)
        t = DataExtraction(c[0], c[1])
        t.get_X_Y() #this function this us (700,260) and (700,) data,label vectors

        self.model = SGDClassifier(loss="log", learning_rate="invscaling", alpha=0.01, eta0=1, penalty="elasticnet",
                                   max_iter=4, tol=1e-4)

        self.X = t.get_X()
        self.y = t.get_Y()
        self.scores_p_a = list()

    def copy(self, x):

        return copy.deepcopy(x)

    def create_and_train_test_folds(self):

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

        for k, (train, test) in enumerate(k_fold.split(X_list, y_list)):
            # create label train and test files
            y_out, y_test = y[train], y[test][0]

            # the dimensions should be reorganized to (560,)
            y_train = np.concatenate(y_out).ravel()

            # split features into train and test groups
            X_test, X_train = X[test][0], X[train]

            # Have to reshape train set from (4,140,260) to (560,260)
            X_train = X_train.reshape((X_train.shape[1] * X_train.shape[0], (X_train.shape[2])))

            # score the p_a accuracy of our DT Classifier
            self.scores_p_a.append(self.model.fit(X_train, y_train).score(X_test, y_test))

            # score the tr_a accuracy of our DT Classifier

            scores_tr_a.append(self.model.fit(X_train, y_train).score(X_train, y_train))

        print("Mean Accuracy when tested on fifth fold. (i.e. p_a)")
        print(self.scores_p_a)
        pa = np.asarray(self.scores_p_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (pa.mean(), pa.std() * 2))

        print("Mean Accuracy when tested on the folds 1-4. (i.e. tr_a)")
        print(scores_tr_a)
        tra = np.asarray(scores_tr_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (tra.mean(), tra.std() * 2))

    def tune_sgd_parameters(self):

        # function for SGD hyper parameter tuning and plotting with PyCharm SciView

        # the values we would like to test for a parameter
        parameter_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004]

        xi = [i for i in range(0, len(parameter_values))]

        parameter_scores = []

        # our original classifier model should be saved if the user wants to run 5-CV Validation with untuned version.
        original_model = self.copy(self.model)

        for param in parameter_values:
            # Let us plot max_leaf_nodes
            self.model = SGDClassifier(loss="log", learning_rate="invscaling", alpha=0.01, eta0=1, penalty="elasticnet",
                                       max_iter=4, tol=param)

            # Perform 5-fold cross validation
            self.create_and_train_test_folds()

            parameter_scores.append(np.asarray(self.scores_p_a).mean())

        self.model = original_model

        plt.plot(xi, parameter_scores, marker='o', linestyle='--', color='r', label='Score')

        plt.xlabel('eta0 values for SGD when learning rate is invscaling')
        plt.ylabel('Cross-validated accuracy')
        plt.xticks(xi, parameter_values)
        plt.legend()

        plt.show()

    def measure_performance(self, clf, X, y, show_accuracy=True, show_classification_report=True,
                            show_confusion_matrix=True):

        clf.fit(self.X,self.y)
        y_pred = clf.predict(X)

        if show_accuracy:
            print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")

        if show_classification_report:
            print("Classification report")
            print(metrics.classification_report(y, y_pred))

        if show_confusion_matrix:
            print("Confusion matrix")
            print(metrics.confusion_matrix(y, y_pred), "\n")

###### USAGE ########

## STEP 1## Create an SGD Object
# First parameter --> path of the train folder
#Second Paremeter --> Path of the fold files (WITHOUT NUMBER!). They must be all under same file directory.
# Third Paremeter --> Number of fold files. Used for k-Fold, Default = 5
sgd = SGD("badges.modified.data.train", "badges.modified.data.fold", 5)

## STEP 2## Cross Validation
# Call this function to get  p_a and tr_a of 5-fold validation
sgd.create_and_train_test_folds()

## STEP 3##  Parameter Tuning
sgd.tune_sgd_parameters() #uncomment this line to experiment with hyperparameters

#STEP 4 If you want to predict a certain fold file uncomment the following 4 lines block
# path = only argument to give is the path of the fold file


c = loadCorpus("badges.modified.data.fold1")
t = DataExtraction(c[0], c[1])
t.get_X_Y()
sgd.measure_performance(sgd.model,t.get_X(),t.get_Y())




