# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:25:16 2024

@author: Maegan Gale C. Prohibido
"""

from scipy.stats import multivariate_normal
import numpy as np

class BayesianLinearRegression:
    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray, noise_var: float):
        self.prior_mean = prior_mean[:, np.newaxis]
        self.prior_cov = prior_cov
        self.prior = multivariate_normal(prior_mean, prior_cov)
        self.noise_var = noise_var
        self.noise_precision = 1 / noise_var
        self.param_posterior = self.prior
        self.post_mean = self.prior_mean
        self.post_cov = self.prior_cov
        

prior_mean = np.array([0, 0])
prior_cov = np.array([[0.5, 0], [0, 0.5]])
noise_var = 0.2
blr = BayesianLinearRegression(prior_mean, prior_cov, noise_var)

import matplotlib.pyplot as plt

def compute_function_labels(slope: float, intercept: float, noise_std_dev: float, data: np.ndarray) -> np.ndarray:
    n_samples = len(data)
    if noise_std_dev == 0:
        return slope * data + intercept
    else:
        return slope * data + intercept + np.random.normal(0, noise_std_dev, n_samples)
    

seed = 42
np.random.seed(seed)

n_datapoints = 1000
intercept = -0.7
slope = 0.9
noise_std_dev = 0.5
noise_var = noise_std_dev**2
lower_bound = -1.5
upper_bound = 1.5


features = np.random.uniform(lower_bound, upper_bound, n_datapoints)
labels = compute_function_labels(slope, intercept, 0., features)
noise_corrupted_labels = compute_function_labels(slope, intercept, noise_std_dev, features)

plt.figure(figsize=(10,7))
plt.plot(features, labels, color='r', label="True values")
plt.scatter(features, noise_corrupted_labels, label="Noise corrupted values")
plt.xlabel("Features")
plt.ylabel("Labels")
plt.title("Real function along with noisy targets")
plt.legend();
