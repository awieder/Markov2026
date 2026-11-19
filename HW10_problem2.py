import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson

lam, mu, T = 12.0, 3.0, 500.0
p = lam / mu
rng = np.random.default_rng(1)

t, n = 0.0, 0
time_in = {}                     
while t < T:
    rate = lam + n * mu
    dwell = rng.exponential(1.0 / rate)
    dwell = min(dwell, T - t)    
    time_in[n] = time_in.get(n, 0.0) + dwell
    t += dwell
    if t >= T:
        break
    n += 1 if rng.random() < lam / rate else -1


kmax = max(time_in)
states = np.arange(kmax + 1)
frac = np.array([time_in.get(k, 0.0) for k in states]) / T
pmf = poisson.pmf(states, p)

diff = np.abs(frac - pmf)
print(f"largest |empirical - Poisson(4)| = {diff.max():.4f} at n = {states[diff.argmax()]}")
print(f"time average of X_t = {np.dot(states, frac):.4f}   (p = {p})")

c = 0
while poisson.sf(c, p) > 0.01:
    c += 1
print(f"smallest c with P(X > c) <= 0.01: c = {c}  (P(X>c) = {poisson.sf(c, p):.5f}, P(X>{c-1}) = {poisson.sf(c-1, p):.5f})")

w = 0.4
plt.figure(figsize=(8, 4.5))
plt.bar(states - w/2, frac, w, label="Simulation (T = 500 h)")
plt.bar(states + w/2, pmf, w, label="Poisson(4)")
plt.xlabel("Bikes out, n"); plt.ylabel("Fraction of time / probability")
plt.title("Bike share: simulated vs stationary distribution")
plt.legend(); plt.tight_layout(); plt.savefig("HW10_problem2.png", dpi=150)