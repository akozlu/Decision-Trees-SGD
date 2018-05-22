import collections
import copy
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import re
from DataExtraction import *
from collections import defaultdict
from io import StringIO
from operator import itemgetter
from sklearn import linear_model, pipeline, feature_extraction, metrics, base
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


# global methods


class DecisionStump(object):

    def __init__(self, training_path, fold_files_path, number_of_fold_files):

        # load all of the data from training path. We do this because we need the dictionary that maps old features to their labels
        self.training_path = training_path
        c = loadCorpus(self.training_path)
        t = DataExtraction(c[0], c[1])
        t.get_X_Y()

        self.X = t.get_X()
        self.y = t.get_Y()

        self.model = SGDClassifier(loss="log", eta0=1e-5)  # create the tuned classifier

        self.fold_files_path = fold_files_path
        self.number_of_fold_files = number_of_fold_files
        self.scores_p_a = list()

        self.old_feature_to_new_feature_dictionary = defaultdict(
            list)  # A dictionary to map old_features (length 260-1D) to new_features (length 100 1-D)
        self.old_feature_to_label_dictionary = t.get_vector_label_dict()
        self.new_feature_to_label_dictionary = {}

    def create_and_score_k_folds(self, test_size):

        # Main function to calculate tr_a and p_a across 5 folds

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
        # Split dataset into 5 consecutive folds

        k_fold = KFold(n_splits=self.number_of_fold_files)

        self.scores_p_a = list()
        scores_tr_a = list()

        for k, (train, test) in enumerate(k_fold.split(X_list, y_list)):

            # create label train and test files
            y_out, y_test = y[train], y[test][0]

            # the dimensions should be reorganized to (560,)
            y_train = np.concatenate(y_out).ravel()

            # split features into train and test groups
            X_test, X_train = X[test][0], X[train]

            # Have to reshape train set from (4,140,260) to (560,260)
            X_train = X_train.reshape((X_train.shape[1] * X_train.shape[0], (X_train.shape[2])))

            # Now we create 100 decision stumps, training them with HALF OF data_train and label_train
            self.create_decision_stump(X_train, y_train, X.reshape((X.shape[1] * X.shape[0], (X.shape[2]))), test_size)

            # this gives us a dictionary that associates old_vectors with new vectors of length 100.

            # Now we create the new dataset to train SGD Classifier
            new_set = self.create_new_vector_label_dataset(self.old_feature_to_new_feature_dictionary)

            # these are the new feature and label set we will training SGD Classifier
            SGD_X = new_set[0]
            SGD_y = new_set[1]

            # Now we transform test_data and test_label to be able to score our SGD

            new_test_data = []
            new_test_label = []

            for x in X_test:
                new_feature = self.old_feature_to_new_feature_dictionary[tuple(x)]
                new_test_data.append(new_feature)
                new_test_label.append(self.old_feature_to_label_dictionary[tuple(x)])

            new_test_data = np.array(new_test_data)
            new_test_label = np.array(new_test_label)

            # We train and score the SGD Classifier on the TEST fold for CALCULATING P_A
            self.scores_p_a.append(self.model.fit(SGD_X, SGD_y).score(new_test_data, new_test_label))

            # We train and score the SGD Classifier on the TRAIN fold for CALCULATING TR_A

            scores_tr_a.append(self.model.fit(SGD_X, SGD_y).score(SGD_X, SGD_y))

            # Before we move to the next fold, we have to clear our dictionary to create 100 1-D vectors again.
            # If you uncomment this line though, you can see how the scores will improve as we get larger length features.

            self.old_feature_to_new_feature_dictionary.clear()

        print("Accuracy when tested on fifth fold. (i.e. p_a)")
        print(self.scores_p_a)
        pa = np.asarray(self.scores_p_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (pa.mean(), pa.std() * 2))

        print("Accuracy when tested on the folds 1-4. (i.e. tr_a)")
        print(scores_tr_a)
        tra = np.asarray(scores_tr_a)
        print("Accuracy: %0.2f (+/- %0.2f)" % (tra.mean(), tra.std() * 2))
        self.old_feature_to_new_feature_dictionary.clear()

    def create_decision_stump(self, X, y, whole_set, test_size):

        # Now we create and train 100 Decision Stumps.
        # Then predict label of each data point in four fold files (560 data points) and store them in a {vector: label list} dictionary

        for i in range(100):

            # randomly sample %50 of our training data and labels
            # data_train and labels_train are %50 of X_train and y_train, which was 4/5 folds. A bit confusing:)

            data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=test_size)

            # train a Decision Stump - Classifier

            clf = tree.DecisionTreeClassifier(max_depth=8)
            clf.fit(data_train, labels_train)

            for data_point in whole_set:
                # for each data point in 5 files predict its label.
                predicted_label = clf.predict(data_point.reshape(1, -1))

                # add the predicted label to the list which is mapped to its data_point
                self.old_feature_to_new_feature_dictionary[tuple(data_point)].append(predicted_label[0])

    def create_new_vector_label_dataset(self, d):

        # Functions associates new 100-1D vectors with their correct labels
        X = []
        y = []
        for old_feature, new_feature in d.items():
            X.append(list(new_feature))
            y.append(self.old_feature_to_label_dictionary[old_feature])

        X = np.array(X)
        y = np.array(y)
        return [X, y]

    def tune_decision_stump_parameters(self):

        # function for Decision Stump parameter tuning. In this example we play with test_size (%20,$03

        # the values we would like to test for a parameter, give example values
        parameter_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

        xi = [i for i in range(0, len(parameter_values))]

        parameter_scores = []

        original_model = self.copy(self.model)

        # our original classifier model should be saved if the user wants to run 5-CV Validation with untuned version.

        for param in parameter_values:
            # Let us plot max_leaf_nodes

            self.model = SGDClassifier(loss="log", eta0=param)  # in this case we are testing values for eta0

            # Perform 5-fold cross validation
            self.create_and_score_k_folds(0.5)

            parameter_scores.append(np.asarray(self.scores_p_a).mean())

        self.model = original_model
        plt.plot(xi, parameter_scores, marker='o', linestyle='--', color='r', label='Score')

        plt.xlabel('Initial Learning Rate for SGD')
        plt.ylabel('Cross-validated accuracy of SGD + Decision Stump')
        plt.xticks(xi, parameter_values)
        plt.legend()

        plt.show()

    def predict_test_data(self, test_path):

        # Function to predict test data. In the end I ended using 8-depth DT classifier instead

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

        # this dictionary maps 260D Vectors to their associated test names
        oldvector_to_test_name_dict = t2.get_test_name_vector_dict()
        test_names_260_to_100_dictionary = defaultdict(list)

        for i in range(100):

            # randomly sample %50 of our training data and labels
            # data_train and labels_train are %50 of X_train and y_train, which was 4/5 folds. A bit confusing:)

            data_train, data_test, labels_train, labels_test = train_test_split(X_train, y_train, test_size=0.5)

            # train a Decision Stump - Classifier

            clf = tree.DecisionTreeClassifier(max_depth=8)
            clf.fit(data_train, labels_train)

            for data_point in X_train:
                # for each data point in training data
                predicted_label = clf.predict(data_point.reshape(1, -1))

                # add the predicted label to the list which is mapped to its 260-D-Vector
                self.old_feature_to_new_feature_dictionary[tuple(data_point)].append(predicted_label[0])

            # in addition we have to create 100-D vectors from the test names.

            for (vector, names) in oldvector_to_test_name_dict.items():
                np_vector = np.array(vector)
                predicted_label = clf.predict(np_vector.reshape(1, -1))

                test_names_260_to_100_dictionary[vector].append(predicted_label[0])
        # this gives us a dictionary that associates old_vectors with new vectors of length 100.

        # Now we have everything we need. A (700,100) data (700,) label to train our model
        # and then test it on (59,100) test set.

        new_set = self.create_new_vector_label_dataset(self.old_feature_to_new_feature_dictionary)

        # these are the new feature and label set we will training SGD Classifier
        SGD_X = new_set[0]
        SGD_y = new_set[1]
        print(SGD_y.shape)
        print(SGD_X.shape)
        self.model.fit(SGD_X, SGD_y)

        for (old_test_vector, new_test_vector) in test_names_260_to_100_dictionary.items():
            np_new_test_vector = np.asarray(new_test_vector)

            print(self.model.predict(np_new_test_vector.reshape(1, -1)))

        self.old_feature_to_new_feature_dictionary.clear()

    def copy(self, x):

        return copy.deepcopy(x)


###### USAGE ########

## STEP 1## Create an Decision Stump Object
# First parameter --> path of the train folder
# Second Paremeter --> Path of the fold files (WITHOUT NUMBER!,example below). They must be all under same file directory.
# Fourth Paremeter --> Number of fold files. Used for k-Fold, Default = 5

DS = DecisionStump("badges.modified.data.train", "badges.modified.data.fold", 5)

## STEP 2## Cross Validation
# param = test size for random sampling. Default is (0.5). We sample randomly from half of test data
# Call this function to get  p_a and tr_a of 5-fold validation
DS.create_and_score_k_folds(0.5)

## STEP 3##  Parameter Tuning
DS.tune_decision_stump_parameters()

##STEP 4##  Test it on test data
# First argument: path of the test set
DS.predict_test_data("badges.modified.data.test")
