#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE_HOEFFDINGTREE
Created on Wed Apr 11 07:28:51 2018

@author: irene
"""

import numbers
import numpy as np
from sklearn import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import Parallel, delayed

SPLIT_FUNCTIONS = {"mean": np.mean, "median": np.median, }

class BaseHoeffdingTree(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 split_criterion="mean",
                 min_samples_split=None,
                 min_samples_split_lower = 2,
                 max_depth=None,
                 random_state=None):

        self.split_criterion = split_criterion

        try:
            self.split_function_ = SPLIT_FUNCTIONS[self.split_criterion]
        except KeyError:
            raise ValueError("Unknown split_criterion {}. Supported crtierions are ``mean``and ``median``".format(self.split_criterion))

        if isinstance(min_samples_split, (numbers.Integral, np.integer)):
            if not 1 <= min_samples_split:
                raise ValueError("min_samples_split must be at least 1, got %s" % min_samples_split)
        elif not isinstance(min_samples_split, type(None)):
            raise ValueError("min_samples_split must be integer or bool, got %s" % min_samples_split)
        self.min_samples_split = min_samples_split

        if not isinstance(min_samples_split_lower, (numbers.Integral, np.integer)):
            raise ValueError("min_samples_split_lower must be integer, got %s" % min_samples_split_lower)
        self.min_samples_split_lower = min_samples_split_lower

        if max_depth is None:
            self.max_depth = (2 ** 31) -1
        else:
            self.max_depth = max_depth
        
        self.random_state = random_state

        self.root_ = None

    def _flush(self):
        for node in self.tree_:
            if node.label is None:
                if sum(node._observed_class_distribution.values()) < 2 and node.parent:
                    node.label = node.parent.label
                else:
                    node.label = max(node._observed_class_distribution, key=lambda k: node._observed_class_distribution[k])
                    if node.label is None:
                        node.label = node.parent.label
        return self
    
    def _prepare_fit_binary(self, X, y):
        # Preprocessing.
        X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
        y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
        
        self._label_encoder = LabelEncoder()
        y = np.array(self._label_encoder.fit_transform(y), dtype=np.int8, order='C')
        # Check if multiclass
        if self._label_encoder.classes_.shape[0] != 2:
            raise ValueError(self.__class__.__name__ + ' only support two class problems')

        check_consistent_length(X, y)

        utils.check_X_y(X, y)
        utils.assert_all_finite(X)

        return X, y

    def fit(self, X, y, sample_weight=None):

        random_state = check_random_state(self.random_state)

        X, y = self._prepare_fit_binary(X, y)

        if y is not None:
            n_samples, self.n_features = X.shape
            self.classes = np.unique(y)
            self.n_classes_ = len(self.classes)
            if sample_weight is None:
                sample_weight = np.ones(n_samples)
            if n_samples != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(n_samples, len(sample_weight)))

            if self.min_samples_split is None:
                min_samples_split = n_samples
            else:
                min_samples_split = self.min_samples_split

            for m in range(min_samples_split, self.min_samples_split_lower, -1):
                self.min_samples_split_ = m

                #for t in range(self.max_samples):
                #    i = random_state.randint(0, n_samples)
                #    if sample_weight[i] != 0.0:
                #        self._partial_fit(X[i], y[i], sample_weight[i])
                
                # Shuffling
                X, y, sample_weight = shuffle(X, y, sample_weight, random_state=random_state)
                for i in range(n_samples):
                    if sample_weight[i] != 0.0:
                        self._partial_fit(X[i], y[i], sample_weight[i])

        self._flush()

        return self
    
    def _traverse_tree(self, X, y=None, weight=None):
        found_node = self.root_
        while not found_node.is_leaf:
            #if y is not None:
            #    try:
            #        found_node._observed_class_distribution[y] += weight
            #    except KeyError:
            #        found_node._observed_class_distribution[y] = weight
            if X[found_node.feature] < found_node.threshold:
                found_node = found_node.left_child
            else:
                found_node = found_node.right_child
        return found_node
    
    def _partial_fit(self, X, y, sample_weight):
        """ Search through the tree until you reach a leaf,
            assign considered sample and split if necessary
        """

        if self.root_ is None:
            self.root_ = TreeNode(depth=1)
            self.tree_ = [self.root_]

        node = self._traverse_tree(X, y, sample_weight)
        node.learn_from_instance(X, y, sample_weight, self.split_function_)
        if node.depth >= self.max_depth:
            node.leaf = True
        elif node._check_split(self.min_samples_split_):
            self.tree_.extend(node._split(self.split_function_))

        return self

    def _predict(self, X):
        return self._traverse_tree(X).label

    def predict(self, X):
        predictions = [self._predict(xi) for xi in X]
        return np.array(predictions)

    def score(self, X, y=None):
        try:
            getattr(self, "tree_")
        except AttributeError:
            raise RuntimeError("You must train classifier before computing score!")

        return(sum(self.predict(X)==y)/len(X))
    
    def is_fitted(self):
        try:
            getattr(self, "tree_")
            return True
        except AttributeError:
            return False

class TreeNode(object):

    def __init__(self,
                 depth=None,
                 parent = None,
                 left_child = None,
                 right_child = None):

        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth

        self.label = None
        self.is_leaf = True # All new nodes are leaves
        #self.is_active = True # All new nodes are active until split is done

        if not hasattr(self, "_threshold"):
            self._threshold = {} # Dictionary (class_value, statistics for each feature)

        if not hasattr(self, "class_observations"): 
            class_observations = {}  # Dictionary (class_value, weight)
        self._observed_class_distribution = class_observations

    def is_left_child(self):
        return self.parent and self.parent.left_child == self

    def is_right_child(self):
        return self.parent and self.parent.right_child == self
    
    def learn_from_instance(self, X, y, weight, split_function):

        try:
            self._threshold[y] = (weight*X + self._observed_class_distribution[y]*self._threshold[y])/(self._observed_class_distribution[y]+weight)
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._threshold[y] = X
            self._observed_class_distribution[y] = weight

        return self
    
    def _check_split(self, min_samples):

        if sum(self._observed_class_distribution.values()) >= min_samples:
            if len(self._observed_class_distribution.keys()) == 1:
                self.label = max(self._observed_class_distribution, key=lambda k: self._observed_class_distribution[k])
                self.is_leaf = True
            else:
                self.is_leaf = False
        else:
            self.is_leaf = True
        return not(self.is_leaf)
    
    def _split(self, split_function):

        candidates = np.stack((self._threshold.values()), 0)
        self.feature = np.argmax(np.amax(candidates, axis=0) - np.amin(candidates, axis=0))
        self.threshold = split_function(candidates[:, self.feature])
        self.left_child = TreeNode(depth=self.depth+1, parent=self)
        self.right_child = TreeNode(depth=self.depth+1, parent=self)

        return self.left_child, self.right_child
