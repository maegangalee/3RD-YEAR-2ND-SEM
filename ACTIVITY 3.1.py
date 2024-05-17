# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:52:00 2024

@author: Maegan Gale
"""

import numpy as np
import matplotlib.pyplot as plt

prior_probs = np.array([[0.33,0.3], [0.2,0.17]])

plt.imshow(prior_probs, cmap='gray')
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.annotate(prior_probs[i,j],(j,i), color="red", fontsize=20, fontweight='bold', ha='center', va='center')
plt.title('Prior Probabilities', fontsize=20)