#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE_HOEFFDINGTREE
Created on Wed Apr 11 07:28:51 2018

@author: irene

TODO:
    REDEFINIR LOS BATCHES
    Para el primer nodo pasan todos a la vez

    GUARDAR LOS PUNTOS
    Hay que guardarse los puntos que pasan por cada nodo para poder hacer el split.
    Una vez hecho el split, se puede eliminar estar información. 

    EVALUAR SPLIT
    Ordenar los puntos por cada feature y probar distintos puntos
    de corte: mediana y percentiles (mediana de la mediana)

    SPLIT
    Calcular la media ponderada por el número de puntos para evaluar la bondad de un
    split teniendo en cuenta la impurity de los nodos hijo
"""

import numbers
import numpy as np
#from abc import ABCMeta, abstractmethod
from sklearn import utils
from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.externals import six
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import Parallel, delayed

#import split_fast as fast

SPLIT_FUNCTIONS = {"mean": np.mean, "median": np.median, }

class BaseHoeffdingTree(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 split_criterion="mean",
                 max_iter=2,
                 min_samples_split=None,
                 min_samples_leaf = 1,
                 min_samples_split_lower = 2,
                 max_depth=None,
                 random_state=None):

        self.split_criterion = split_criterion

        try:
            self.split_function_ = SPLIT_FUNCTIONS[self.split_criterion]
        except KeyError:
            raise ValueError("Unknown split_criterion {}. Supported crtierions are ``mean``and ``median``".format(self.split_criterion))

        if not isinstance(max_iter, (numbers.Integral, np.integer)):
            raise ValueError("max_iter must be integer, got %s" % max_iter)
        self.max_iter = max_iter

        if not isinstance(min_samples_leaf, (numbers.Integral, np.integer)):
            raise ValueError("min_samples_leaf must be integer, got %s" % min_samples_leaf)
        self.min_samples_leaf = min_samples_leaf

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

    def is_fitted(self):
        try:
            getattr(self, "tree_")
            return True
        except AttributeError:
            return False

    def _flush(self):
        for node in self.tree_:
            if node.label is None:
                if sum(node._observed_class_distribution.values()) < 2 and node.parent:
                    node.label = node.parent.label
                else:
                    node.label = max(node._observed_class_distribution, key=lambda k: node._observed_class_distribution[k])
                    if node.label is None:
                        node.label = node.parent.label 
            #if node.label is None:
            #    if sum(node.classes_) < 2 and node.parent:
            #        node.label = node.parent.label
            #    else:
            #        node.label = node.classes[np.argmax(node.classes_)]
            #        if node.label is None:
            #            node.label = node.parent.label 
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
            #if (self.max_samples is None) or (self.max_samples > n_samples):
            #    max_samples = n_samples
            if self.max_iter is None:
                max_iter = 1
            else:
                max_iter = self.max_iter

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
                for it in range(max_iter):
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
            #    node.classes_[node.classes == y] += 1
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
            #self.tree_.extend(node._split(self.split_function_))
            idx, thr = self._best_split(node.X, node.y)
            #idx, thr = fast.best_split(node.X.astype('float'), node.y.astype('float'), self.n_classes_, self.n_features)
            if idx == None:
                node.is_leaf = True
            else:
                self.tree_.extend(node._split_node(idx, thr))

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
    
    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        
        #m = len(y)
        m = X.shape[0]

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        #num_parent = node._observed_class_distribution

        # Gini of current node.
        #best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_misclassification = 1.0 - sum((n / m) for n in num_parent if n != max(num_parent))
        #best_misclassification = 1.0 - sum(n/m for k,n in num_parent.items() if k != max(num_parent, key=lambda k: num_parent[k]))
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            #num_left = dict.fromkeys(num_parent.keys(), 0)
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                #gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                #gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                misclassification_left = 1.0 - sum((n / i) for n in num_left if n != max(num_left))
                misclassification_right = 1.0 - sum((n / (m-i)) for n in num_right if n != max(num_right)) 

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                #gini = (i * gini_left + (m - i) * gini_right) / m
                misclassification = (i * misclassification_left + (m - i) * misclassification_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                #if gini < best_gini:
                #    best_gini = gini
                #    best_idx = idx
                #    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

                if misclassification < best_misclassification:
                    best_misclassification = misclassification
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

class TreeNode(object):

    def __init__(self,
                 class_observations = None,
                 depth=None,
                 parent = None,
                 left_child = None,
                 right_child = None):
        
        if class_observations is None:
            class_observations = {}  # Dictionary (class_value, weight)
        self._observed_class_distribution = class_observations
        
        self.depth = depth
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

        self.label = None
        self.is_leaf = True # All new nodes are leaves

        self._threshold = {} # Dictionary (class_value, statistics for each feature)
    
    def is_left_child(self):
        return self.parent and self.parent.left_child == self

    def is_right_child(self):
        return self.parent and self.parent.right_child == self
    
    def learn_from_instance(self, X, y, weight, split_function):

        try:
            #self._threshold[y] = split_function((weight*X, self._observed_class_distribution[y]*self._threshold[y]), axis=0)
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
        
        try:
            self.X = np.vstack((self.X, X))
            self.y = np.append(self.y, y)
        except AttributeError:
            #self._threshold[y] = X
            #self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))
            self.X = X
            self.y = y

        #if self.classes_[self.classes == y] == 0:
        #    self.thr_[y] = X
        #else:
        #    self.thr_[y] = split((X, self.thr_[y]), axis=0)
        #self.classes_[self.classes == y] += 1

        return self
    
    def _check_split(self, min_samples):

        #if sum(self._observed_class_distribution.values()) >= min_samples:
        if self.y.size >= min_samples:
            #if len(self._observed_class_distribution.keys()) == 1:
            if len(np.unique(self.y)) == 1:
                self.label = np.unique(self.y)[0] 
                #self.label = max(self._observed_class_distribution, key=lambda k: self._observed_class_distribution[k])
            #if sum(self.classes_ == 0) == self.n_classes - 1:
                #self.label = self.classes[np.argmax(self.classes_)]
                self.is_leaf = True
            else:
                self.is_leaf = False
        else:
            self.is_leaf = True
        return not(self.is_leaf)

    def _split_node(self, idx, thr):

        self.feature, self.threshold = idx, thr
        #candidates = np.stack((self._threshold.values()),1)
        #self.feature = np.argmax(np.amax(candidates, axis=0) - np.amin(candidates, axis=0))
        #self.threshold = split_function(candidates[:, self.feature])
        self.left_child = TreeNode(depth=self.depth+1, parent=self)
        self.right_child = TreeNode(depth=self.depth+1, parent=self)

        # Delete memory once split has been perfomed
        del self.X
        del self.y

        return self.left_child, self.right_child
    
    '''
    def _split(self, split_function):

        candidates = np.stack((self._threshold.values()),1)
        self.feature = np.argmax(np.amax(candidates, axis=0) - np.amin(candidates, axis=0))
        self.threshold = split_function(candidates[:, self.feature])
        self.left_child = TreeNode(depth=self.depth+1, parent=self)
        self.right_child = TreeNode(depth=self.depth+1, parent=self)

        return self.left_child, self.right_child
    '''

    
