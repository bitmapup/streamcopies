#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMPLER

Created on Thu Jan 25 16:25:12 2018

@author: irene

Different sampling strategies
to query pre-traiend models and obtain new
synthetic datasets.
"""

import random
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from base_Sampler import BaseSampler



class ExpandAndContract(BaseSampler):

    """ Expand and contract sampler.

        Sample the class probability space of a trained model. Do so
        by first making a large random query to try and conver the most
        part of the space and then prune to obtain a balance synthetic
        dataset.

        Parameters
        ----------
        max_queries : int, optional (default=1000)
            Total number of queries. Equald the size of the synthetic dataset.

        n_exploit : int, optional (default=10)
            Number of queries per exploitation loop. Each loop corresponds to either an exploration
            or an exploitation step.

        n_explore : int, optional (default=10)
            Number of queries per exploration loop. Each loop corresponds to either an exploration
            or an exploitation step.

        n_batch : int, optional (default=10)
            Number of batches for the exploration step.

        min_ratio : float, optional (default=0.1)
            Maximum deviation allowed over balanced distributions of samples in the
            synthetic dataset. For a completely balanced dataset, set to 0.

        step : float, optional (default=1e-3)
            Radius of the search sphere during the exploitation step. At each step
            the radius step is adjusted to obtain the desired samples.
        """

    def __init__(self,
                  max_queries=1000,
                  n_exploit=10,
                  n_explore=100000,
                  n_batch=100,
                  step=1e-3,
                  random_state=None,
                  verbose=False):

        super(ExpandAndContract, self).__init__(max_queries=max_queries,
                                            n_exploit=n_exploit,
                                            n_explore=n_explore,
                                            n_batch=n_batch,
                                            step=step,
                                            random_state=random_state,
                                            verbose=verbose)

    def increase_sampling_size(n_explore, buffer, max_queries, n_classes, logger):
	    """ When the region corrresponding to a certain class is very small,
		it takes a lot of time to
		gather all the required samples. Increase the sampling size.
	    """
	    if any(buffer < 0.1*max_queries/n_classes) and n_explore < 10000000:
			logger.info('Incresing sample size by a 10-fold')
			return 10*n_explore
	    else:
			return n_explore

    def get_seeds(estimator, X, y, classes, buffer, n_exploit, logger):
	    print ("get_seeds")
	    over_sampled = classes[buffer > 0.]
	    sub_sampled = classes[buffer == 0.]
	    logger.info('Oversampled classes: {}'.format(over_sampled))
	    logger.info('Subsampled classes: {}'.format(sub_sampled))

	    # La referencia no deben ser las samples originales, sino la predicción que hace el oracle sobre ellas
	    masker = np.isin(estimator.predict(X), sub_sampled).ravel()

	    # TODO aquí hay que decidir si se cogen todas las samples originales o no
	    # si es que si, no tiene sentido usar el min
	    # n_min = self.n_exploit
	    n_min = len(estimator.predict(X)[masker])
	    logger.info('Seeds that will be extracted: {}'.format(min(n_min, len(estimator.predict(X)[masker]))))

	    if min(n_min, len(estimator.predict(X)[masker])) == 0.:
		logger.info('There is at least one class that has not been learnt by the original model')
		return None, None, None , True, np.unique(np.unique(estimator.predict(X))), len(np.unique(np.unique(estimator.predict(X))))

	    samples = np.vstack(random.sample(list(X[masker]), min(n_min, len(estimator.predict(X)[masker]))))
	    labels = estimator.predict(samples)

	    return samples, labels, sub_sampled, False, classes, len(classes)

    def search_neighborhood(sub_sampled, xi, estimator, mins, maxs, n, step, n_numeric, n_categorical, logger):
	    """ Search additional samples in the vicinity of the provided point.

	    Parameters :
	    ------------

		xi : numpy array sample
		n  : number of rows
	    """
	    logger.info('Searching neighborhood with initial step {}'.format(step))
	    done = False
	    while not done:
		# Generate random direction vector and step size
		samples = np.tile(xi, (n, 1))
		v = np.random.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), n)
		h = np.random.uniform(0, step, size=n)
		samples[:, :n_numeric] = samples[:, :n_numeric] + h[:,np.newaxis]*v

		# Check new points are within boundaries. If not, reduce step size.
		# Only exit if all new points belong to the desired class
		labels = estimator.predict(samples)
		if (samples[:, :n_numeric] >= mins).all() and (samples[:, :n_numeric] <= maxs).all() and (np.isin(labels, sub_sampled)).all():
		    logger.info('All new samples belong to the right classes')
		    done = True
		else:
		    logger.info('Sampled labels: {}'.format(labels))
		    logger.info('Sub_sampled labels: {}'.format(sub_sampled))
		    logger.info('Decreasing step size')
		    step = 0.1*step
	    return samples, labels

    def resample(max_queries, estimator, X, y, n_dim, n_classes_, classes_, logger, cols=None):
        samples = np.empty((0, n_dim)) if not isinstance(X, pd.DataFrame) else pd.DataFrame(columns=cols)
        for i in classes_:
	    _samples = X[y == i]
	    samples = np.append(samples, _samples[np.random.choice(_samples.shape[0], int(max_queries/n_classes_), replace=False), :], axis=0) if not isinstance(X, pd.DataFrame) else samples.append(_samples.sample(n=int(max_queries/n_classes_), replace=False))
        logger.info('Resampled!')
        return samples




    def _compute_buffer(self):
        counts = np.bincount(self.y_, minlength=self.n_classes_)
        if self.shrink_:
            nonzero = np.nonzero(counts)[0]
            return counts[nonzero]
        else:
            return counts

    def _explore(self):
        self.logger.info('Exploring')
        self.n_explore = increase_sampling_size(self.n_explore, self.buffer, self.max_queries, self.n_classes_, self.logger)

        # While any given class is subsampled with respect to a given threshold, continue to explore the space randomly (this ensures that we do not
        # focus on certain regions of the space, but rather explore the full area with an equal probability)
        samples, labels = self._make_random_queries()
        print ("samples ",samples)
        # Only store in memory samples that are meaningful
        masker = np.isin(labels, self.classes_[self.buffer < int(self.max_queries/self.n_classes_)])
        print ('labels: ',labels)

        print ('self.buffer: ',self.buffer)
        print ('int(self.max_queries/self.n_classes_): ',int(self.max_queries/self.n_classes_))
        print ('self.classes_',self.classes_)
        print ('masker',masker)
        samples = samples[masker]
        labels = labels[masker]

        self.X_ = np.append(self.X_, samples, axis=0) if not isinstance(self.X, pd.DataFrame) else self.X_.append(samples)
        self.y_ = np.append(self.y_, labels)
        self.n_queries += len(samples)
        self.buffer = self._compute_buffer()
        self.logger.info('Resulting buffer: {}'.format(self.buffer))

    def _exploit_classes(self):
        """ Search new samples of the minority classes.
            """
        self.logger.info('Exploiting seeded classes')
        masker = np.isin(self.y_, self.sub_sampled).ravel()

        results = list(zip(*(Parallel(n_jobs=1)(delayed(search_neighborhood)(self.sub_sampled, xi, self.oracle, self.mins_, self.maxs_, self.n_exploit, self.step, n_numeric_, n_categorical_, self.logger) for xi in self.X_[masker]))))

        samples = np.concatenate(results[0])
        labels = np.concatenate(results[1])

        self.X_ = np.append(self.X_, samples, axis=0) if not isinstance(self.X, pd.DataFrame) else self.X_.append(samples)
        self.y_ = np.append(self.y_, labels)
        self.n_queries += len(samples)
        self.buffer = self._compute_buffer()
        self.logger.info('Resulting buffer: {}'.format(self.buffer))
    ## AQUI
    def _query(self):
        print ("ExpandAndContract(BaseSampler) ", " _query")
        # Start by randomly searching the whole class probability space
        self.logger.info('Initial exploration')
        self.X_, self.y_ = self._make_random_queries()
        self.n_queries += self.n_explore
        self.buffer = self._compute_buffer()
        self.logger.info('Initial buffer: {}'.format(self.buffer))

        runs = 0
        while any(self.buffer < int(self.max_queries/self.n_classes_)):
            if all(self.buffer > 0.) or runs < 2:
                self._explore()
                runs += 1

                if self.n_seeds_ > 0:
                    if any(self.buffer[self.sub_sampled] < int(self.max_queries/self.n_classes_)):
                        # Exploit seeded classes
                        self._exploit_classes(self.classes_[self.sub_sampled][self.buffer[self.sub_sampled] < int(self.max_queries/self.n_classes_)], X, y)

            else:
                # If after two iterations no samples are found for one or more classes, allow knowledge of the original
                # dataset and include randomly chosen training points
                self.logger.info('Seeds are needed')
                samples, labels, self.sub_sampled, self.shrink_, self.classes_, self.n_classes_ = get_seeds(self.oracle, self.X, self.y, self.classes_, self.buffer, self.n_exploit, self.logger)

                if self.shrink_:
                    self.logger.info('Shrinked exploration')
                    self.X_, self.y_ = self._make_random_queries()
                    self.n_queries += self.n_explore
                    self.buffer = self._compute_buffer()
                    self.logger.info('Resulting buffer: {}'.format(self.buffer))
                    continue

                self.n_seeds_ += len(labels)

                self.X_ = np.append(self.X_, samples, axis=0) if not isinstance(self.X, pd.DataFrame) else self.X_.append(samples)
                self.y_ = np.append(self.y_, labels)
                self.buffer = self._compute_buffer()
                self.logger.info('Seeded buffer: {}'.format(self.buffer))

                self.n_queries += len(samples)

        # Once a minimum number of samples for each class is obtained, re-sample to obtain a balanced synthetic dataset
        self.logger.info('Resampling to adjust population sizes')
        samples = resample(self.max_queries, self.oracle, self.X_, self.y_, self.n_dim_, self.n_classes_, self.classes_, self.logger, cols=None if not isinstance(self.X, pd.DataFrame) else self.cols_)

        self.X_ = samples
        if isinstance(self.oracle, BaseEstimator):
            self.y_ = self.oracle.predict(samples).astype(int)
        else:
            self.y_ = np.argmax(self.oracle.predict(samples), axis=1).astype(int)
            self.n_queries += len(samples)
            self.buffer = self._compute_buffer()

        return self.X_, self.y_
