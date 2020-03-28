#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE_HOEFFDINGTREE
Created on Wed Apr 11 07:28:51 2018

@author: irene
"""

import numbers
import numpy as np
#from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.externals import six
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

SPLIT_FUNCTIONS = {"mean": np.mean, "median": np.median, }

class BaseHoeffdingTree(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 splitter="mean",
                 min_samples_split=100,
                 max_samples=1000,
                 min_samples_leaf = 1,
                 min_samples_split_lower = 2,
                 max_depth=None,
                 random_state=None):

        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split_lower = min_samples_split_lower
        self.max_depth = max_depth
        self.random_state = random_state

    def _get_split_function(self, split):
        """ Get concrete ``SplitFunction`` object for str ``split``.
        """
        try:
            split_ = SPLIT_FUNCTIONS[split]
            return split_
        except KeyError:
            raise ValueError("Unknown splitter value {}. Supported splitters are ``mean``and ``median``".format(split))

    def _traverse_tree(self, X, y=None, fitted=True):
        node = self.root_
        depth = 0
        while not node.leaf:
            if not fitted:
                node.classes_[node.classes == y] += 1
            if X[node.axis] < node.thr:
                node = node.left_child
            else:
                node = node.right_child
            depth += 1
        return node, depth

    def _prune(self, node):
        # If node count smaller than min_sample_leaf,
        # remove children and redefine node as leaf
        if not node.leaf and sum(node.classes_) < self.min_samples_leaf:
            self.tree_.remove(node.left_child)
            self.tree_.remove(node.right_child)
            self._prune(self, node.left_child)
            self._prune(self, node.right_child)
            node.left_child = None
            node.right_child = None
            node.leaf = True
        return self

    def _flush(self):
        for node in self.tree_:
            if node.label is None:
                if sum(node.classes_) < 2 and node.parent:
                    node.label = node.parent.label
                else:
                    node.label = node.classes[np.argmax(node.classes_)]
                    if (node.label = None):
                        node.label = node.parent.label
                    #node.label = node.classes[np.argmax(node.classes_)]
        return self

    def _partial_fit(self, X, y, max_depth):
        """ Search through the tree until you reach a leaf,
            assign considered sample and split if necessary
        """
        node, depth = self._traverse_tree(X, y, fitted=False)
        node.update(X, y, self.split_function_)
        if depth >= max_depth:
            node.leaf = True
        elif node._check_split(self.min_samples_split_):
            self.tree_.extend(node._split(self.split_function_))

        return self

    def fit(self, X, y):
        # Preprocessing.
        X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
        y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
        check_consistent_length(X, y)

        n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(np.unique(y))

        # Define seed for random number generator
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        np.random.seed(seed)

        # Define splitter
        try:
            self.split_function_ = SPLIT_FUNCTIONS[self.splitter]
        except KeyError:
            raise ValueError("Unknown splitter value {}. Supported splitters are ``mean``and ``median``".format(self.splitter))

        # Check parameters
        max_depth = ((2 ** 31) -1 if self.max_depth is None else self.max_depth)

        if not isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            raise ValueError("min_samples_leaf must be integer, got %s" % self.min_samples_leaf)

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            if not 1 <= self.min_samples_split:
                raise ValueError("min_samples_leaf must be at least 1, got %s" % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:
            raise ValueError("min_samples_leaf must be integer, got %s" % self.min_samples_split)

        # Initialize tree to root node
        self.root_ = HNode(classes=self.classes_, n_features=self.n_features)
        self.tree_ = [self.root_]

        if self.max_samples is None:
            self.max_samples = int(np.ceil(n_samples/2))

        for m in range(min_samples_split, self.min_samples_split_lower, -1):
            self.min_samples_split_ = m
            for t in range(self.max_samples):
                idx = np.random.randint(0, n_samples)
                xi, yi = X[idx], y[idx]
                self._partial_fit(xi, yi, max_depth)

        #self._prune()
        self._flush()

        return self

    def partial_fit(self, X, y):
    
        # Preprocessing.
        X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
        y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
        check_consistent_length(X, y)

        n_samples, self.n_features = X.shape
        
        for m in range(self.min_samples_split, self.min_samples_split_lower, -1):
            self.min_samples_split_ = m
            for t in range(self.max_samples):
                idx = np.random.randint(0, n_samples)
                xi, yi = X[idx], y[idx]
                self._partial_fit(xi, yi, self.max_depth_)

        #self._prune()
        self._flush()

        return self

    def _predict(self, X):
        return self._traverse_tree(X)[0].label

    def predict(self, X):
        predictions = [self._predict(xi) for xi in X]
        return np.array(predictions)

    def score(self, X, y=None):
        try:
            getattr(self, "tree_")
        except AttributeError:
            raise RuntimeError("You must train classifier before computing score!")

        return(sum(self.predict(X)==y)/len(X))

class HNode():

    def __init__(self,
                 classes,
                 n_features,
                 leaf = True,
                 axis = None,
                 thr = None,
                 parent = None,
                 left_child = None,
                 right_child = None,
                 label=None):

        self.classes = classes
        self.n_features = n_features
        self.leaf = leaf
        self.axis = axis
        self.thr = thr
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.label = label

        self.n_classes = len(self.classes)

        if not hasattr(self, "thr_"):
            self.thr_ = np.zeros((self.n_classes, self.n_features))

        if not hasattr(self, "classes_"):
            self.classes_ = np.zeros((self.n_classes))

    def is_left_child(self):
        return self.parent and self.parent.left_child == self

    def is_right_child(self):
        return self.parent and self.parent.right_child == self

    def _check_split(self, min_samples):
        if sum(self.classes_) >= min_samples:
            if sum(self.classes_ == 0) == self.n_classes - 1:
                self.label = self.classes[np.argmax(self.classes_)]
                self.leaf = True
            else:
                self.leaf = False
        else:
            self.leaf = True
        return not(self.leaf)

    def _split(self, splitter):
        self.axis = np.argmax(np.amax(self.thr_, axis=0) - np.amin(self.thr_, axis=0))
        self.thr = splitter(self.thr_[:, self.axis])
        self.left_child = HNode(classes=self.classes, n_features=self.n_features, parent=self)
        self.right_child = HNode(classes=self.classes, n_features=self.n_features, parent=self)

        return self.left_child, self.right_child

    def update(self, X, y, split):
        if self.classes_[self.classes == y] == 0:
            self.thr_[y] = X
        else:
            self.thr_[y] = split((X, self.thr_[y]), axis=0)
        self.classes_[self.classes == y] += 1

        return self
