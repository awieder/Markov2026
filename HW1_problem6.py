import numpy as np
import matplotlib.pyplot as plt
N_vals = [10, 100, 1000, 10000, 100000, 1000000, 10**7, 10**8]
estimates = []
for n in N_vals:
    X = np.random.uniform(0, 1, n)
    Y = np.random.uniform(0, 1, n)
    Z = np.random.uniform(0, 1, n)
    success = (X**2 + Y**2 < Z) & (Z**2 > X*Y)
    estimate = np.mean(success)

    estimates.append(estimate)


for N, estimate in zip(N_vals, estimates):
    print("N =", N, "Monte Carlo estimate =", estimate)

# Plot
plt.plot(N_vals, estimates, marker='o', label='Monte Carlo estimate')

plt.xscale('log')

plt.xlabel('Sample size N')
plt.ylabel('Estimated probability')
plt.title('Monte Carlo Estimate vs Sample Size')
plt.legend()
plt.grid(True)

plt.show()