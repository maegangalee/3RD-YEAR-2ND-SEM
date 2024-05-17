import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior parameters
alpha_prior = 2
beta_prior = 2

# Data from the clinical trial
n_trials = 50
n_successes = 30

# Define the prior distribution
prior_distribution = beta(alpha_prior, beta_prior)

# Update the prior using Bayes' theorem
alpha_posterior = alpha_prior + n_successes
beta_posterior = beta_prior + n_trials - n_successes
posterior_distribution = beta(alpha_posterior, beta_posterior)

# Generate points to plot the prior and posterior distributions
x = np.linspace(0, 1, 1000)
prior_pdf = prior_distribution.pdf(x)
posterior_pdf = posterior_distribution.pdf(x)

# Plot the prior and posterior distributions
plt.figure(figsize=(10, 6))
plt.plot(x, prior_pdf, label='Prior', color='blue')
plt.plot(x, posterior_pdf, label='Posterior', color='red')
plt.xlabel('Success Rate (p)')
plt.ylabel('Probability Density')
plt.title('Prior and Posterior Distributions')
plt.legend()
plt.show()
