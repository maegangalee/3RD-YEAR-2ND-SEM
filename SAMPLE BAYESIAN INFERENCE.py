import numpy as np
from scipy.integrate import quad

# Define the likelihood function
def likelihood(p, k, n):
    return p**k * (1-p)**(n-k)

# Define the uniform prior
def prior(p, n):
    return 1 / (n + 1)

# Define the posterior distribution
def posterior(p, k, n):
    return likelihood(p, k, n) * prior(p, n)

# Parameters
k = 6  # Number of heads observed
n = 10  # Total number of flips

# Calculate the normalization constant (marginal likelihood)
norm_constant, _ = quad(posterior, 0, 1, args=(k, n))

# Calculate the posterior distribution
def posterior_normalized(p):
    return posterior(p, k, n) / norm_constant

# Calculate the posterior distribution over the range [0, 1]
p_values = np.linspace(0, 1, 1000)
posterior_values = [posterior_normalized(p) for p in p_values]

# Plot the posterior distribution
import matplotlib.pyplot as plt
plt.plot(p_values, posterior_values)
plt.xlabel('Probability of heads (p)')
plt.ylabel('Posterior probability density')
plt.title('Posterior distribution after observing 6 heads out of 10 flips')
plt.show()
