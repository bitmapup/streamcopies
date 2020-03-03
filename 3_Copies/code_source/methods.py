#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONLINE COPIES

Created on Thu Dec 14 16:33:45 2018

@author: irene
"""

import time
import random
import itertools
import scipy.spatial.distance
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_consistent_length

from methods.base_HoeffdingTree import BaseHoeffdingTree

import sys
sys.path.insert(0, '/Users/irene/Documents/Projects/dual/scripts/methods/')

from base_HoeffdingTree import BaseHoeffdingTree
#from base_OnlineSVM import BaseOnlineSVM
#from base_CopyOnlineSVM import BaseCopyOnlineSVM

class HoeffdingTree(BaseHoeffdingTree):
    """ A Hoeffding tree classifier.

        Parameters
        ----------
        splitter : string, optional (default="mean")
            Strategy used to choose the split at each node. Supported
            strategies are "mean" to choose the mean split and "median" to choose
            the median split.

        min_samples_split: int, optional (default=100)
            Minimum number of samples required to split an internal node. In
            practice, it corresponde to the maximum number of minimum samples to
            split a node, since this value is decreased during each iteration.

        max_samples : int, optional (default=1000)
            Number of random samples drawn during each iteration of the method.
            Number of iterations to perform.
        
        max_depth : int or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves ocntain less than min_samples_split
            samples.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        """

    def __init__(self,
                 splitter="mean",
                 min_samples_split=100,
                 max_samples=1000,
                 min_samples_leaf = 1,
                 min_samples_split_lower = 2,
                 max_depth=None,
                 random_state=None):

        super(HoeffdingTree, self).__init__(splitter=splitter,
                                            min_samples_split=min_samples_split,
                                            max_samples=max_samples,
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split_lower=min_samples_split_lower,
                                            max_depth=max_depth,
                                            random_state=random_state)

    def fit(self, X, y):
        """ Build a Hoeffding tree classifier from the training set (X, y).

            For each iteration t, evaluate a randomly selected
            set of samples of size equal to max_samples. Expand the tree until samples
            are classified and update internal parameters.

            Parameters
            ----------
            X : array-like or sparse matrix, shape = [n_samples, n_features]
                Set of samples, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
                Set of labels, where n_samples is the number of samples.

            Returns
            -------
            self : object
                Returns self.
        """

        super(HoeffdingTree, self).fit(X, y)

        return self
    
    def predict_proba(self, X):
        """ Compute a class probability estiamtor.

            Parameters
            ----------
            X : array-like or sparse matrix, shape = [n_samples, n_features]
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features. Original data points
            (used to define feature domains).

            Returns
            -------
            p : array of shape = [n_samples]
                The class probabilities of the input samples.
            """
        X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
        all_proba = []
        for xi in X:
            node = self._traverse_tree(xi)[0]
            all_proba.append(node.classes_/sum(node.classes_))
        return np.array(all_proba)

class OnlineSVM(BaseOnlineSVM):
    """ Online kernel SVM.

        Trains a support vector machine method with a custom gaussian
        kernel on an incremental fashion. It does so by using stochastic gradient
        descent to evaluate a randomly chosen set of points at each iteration
        until either convergence or the mximum number of iterations are reached

        Parameters
        ----------
        n_iter : int, optional (default=1000)
            Number of iterations to perform. The process may stop earlier,
            provided convergence is reached.

        C : float, optional (default=100)
            Regularization parameter. Defines the penalty of the error term for
            misclassified samples.

        gamma : float, optional (default=10)
            Kernel coefficient for RBF. Corresponds to
            depth of the gaussian kernel.

        penalty : string, optional (default='2')
            Penalty norm for computed support vectors. Evaluation of the instantaneous
            loss function and its gradient will be computed accordingly.

        optimize_gamma : bool, optional (default=False)
            If True, optimize gamma parameter internally.

        max_samples : int, optional (default=100)
            Number of random samples drawn during each iteration of the method.

        tol: float, optional (default=1e-4)
            Tolerance for convergence.

        gamma_tol: float, optional (default=1e-5)
            Tolerance for convergence of the gamma parameter.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
    """

    def __init__(self,
                 n_iter = 10000,
                 C=100,
                 gamma=10,
                 penalty='l2',
                 optimize_gamma=False,
                 max_samples=100,
                 tol=1e-4,
                 gamma_tol=1e-5,
                 random_state=None):

        super(OnlineSVM, self).__init__(n_iter = n_iter, C = C, gamma = gamma,
                                  penalty = penalty, optimize_gamma = optimize_gamma, max_samples = max_samples,
                                  tol = tol, gamma_tol = gamma_tol, random_state=random_state)

    def fit(self, X, y=None):
        """ Build an Online kernel SVM from the training set (X,y).

            For each iteration t, evaluate a randomly selected
            set of samples of size equal to max_samples. Update the parameters
            sequentially. Once each iteration is completed, evaluated the loss
            and proceed to the next loop. Do so until convergence.

            Parameters
            ----------

            X : array-like or sparse matrix, shape = [n_samples, n_features]
                Set of samples, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
                Set of labels, where n_samples is the number of samples.

            Returns
            -------
            self : object
                Returns self.
            """

        super(OnlineSVM, self).fit(X, y)

        return self

    def decision_function(self, X):
        return self.project(X)

    def score(self, X, y=None):
        try:
            getattr(self, "eta_")
        except AttributeError:
            raise RuntimeError("You must train classifier before computing score!")
        return(sum(self.predict(X)==y)/len(X))

class CopyOnlineSVM(BaseCopyOnlineSVM):
    """ Copy Online kernel SVM.
        
        Trains a support vector machine method with a custom gaussian
        kernel on an incremental fashion. It does so by using stochastic gradient
        descent to evaluate a randomly chosen set of points at each iteration
        until either convergence or the mximum number of iterations are reached
        
        Parameters
        ----------
        n_iter : int, optional (default=1000)
        Number of iterations to perform. The process may stop earlier,
        provided convergence is reached.
        
        C : float, optional (default=100)
        Regularization parameter. Defines the penalty of the error term for
        misclassified samples.
        
        gamma : float, optional (default=10)
        Kernel coefficient for RBF. Corresponds to
        depth of the gaussian kernel.
        
        penalty : string, optional (default='2')
        Penalty norm for computed support vectors. Evaluation of the instantaneous
        loss function and its gradient will be computed accordingly.
        
        optimize_gamma : bool, optional (default=False)
        If True, optimize gamma parameter internally.
        
        max_samples : int, optional (default=100)
        Number of random samples drawn during each iteration of the method.
        
        tol: float, optional (default=1e-4)
        Tolerance for convergence.
        
        gamma_tol: float, optional (default=1e-5)
        Tolerance for convergence of the gamma parameter.
        
        random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        """
    
    def __init__(self,
                 n_iter = 10000,
                 C=100,
                 gamma=10,
                 penalty='l2',
                 optimize_gamma=False,
                 max_samples=100,
                 tol=1e-4,
                 gamma_tol=1e-5,
                 random_state=None):
        
        super(CopyOnlineSVM, self).__init__(n_iter = n_iter, C = C, gamma = gamma,
                                        penalty = penalty, optimize_gamma = optimize_gamma, max_samples = max_samples,
                                        tol = tol, gamma_tol = gamma_tol, random_state=random_state)
    
    def fit(self, X, y=None):
        """ Build an Online kernel SVM from the training set (X,y).
            
            For each iteration t, evaluate a randomly selected
            set of samples of size equal to max_samples. Update the parameters
            sequentially. Once each iteration is completed, evaluated the loss
            and proceed to the next loop. Do so until convergence.
            
            Parameters
            ----------
            
            X : array-like or sparse matrix, shape = [n_samples, n_features]
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
            
            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Set of labels, where n_samples is the number of samples.
            
            Returns
            -------
            self : object
            Returns self.
            """
        
        super(CopyOnlineSVM, self).fit(X, y)
        
        return self
    
    def decision_function(self, X):
        return self.project(X)
    
    def score(self, X, y=None):
        try:
            getattr(self, "eta_")
        except AttributeError:
            raise RuntimeError("You must train classifier before computing score!")
        return(sum(self.predict(X)==y)/len(X))
