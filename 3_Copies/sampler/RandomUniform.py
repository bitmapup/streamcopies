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


class RandomUniform(BaseSampler):

    """ Random uniform sampler.

        Define an oracle to make uniformly distributed queries to the primary model and
        obtain predicted class labels.

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

        min_ratio : float, optional (default=0.1)
            Maximum deviation allowed over balanced distributions of samples in the
            synthetic dataset. For a completely balanced dataset, set to 0

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

        super(RandomUniform, self).__init__(max_queries=max_queries,
                                            n_exploit=max_queries,
                                            n_explore=max_queries,
                                            n_batch=n_batch,
                                            step=step,
                                            random_state=random_state,
                                            verbose=verbose)

    def _query(self):
        print ("RandomUniform(BaseSampler) ", "_query")
        return self._make_random_queries()

