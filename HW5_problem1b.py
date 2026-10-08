import numpy as np
import matplotlib.pyplot as plt
a,b = 1.0, 1.0
p = np.zeros((5,5))
for k in range(5):
    if k < 4:
        p[k, k+1] = a * (4 - k) / 4
    if k > 0:
        p[k, k-1] = b * k / 4
    p[k, k] = 1 - p[k].sum()      

assert np.allclose(p.sum(axis=1), 1)   # rows must sum to 1

N = 60
q = np.zeros(5); q[0] = 1.0
Q = np.zeros((N + 1, 5))
Q[0] = q
for n in range(1, N + 1):
    q = q @ p                         
    Q[n] = q

print("q50 =", Q[50])
print("q51 =", Q[51])

n = np.arange(N+1)
run_avg = np.cumsum(Q[:, 2]) / (n+1)
#Plotting
plt.figure(figsize=(8, 4.5))
plt.plot(n, Q[:, 2], 'o-', ms=4, label=r'$q_n(2)$')
plt.plot(n, Q[:, 4], 's-', ms=4, label=r'$q_n(4)$')
plt.plot(n, run_avg, 'k-', lw=2, label=r'running avg $\frac{1}{n+1}\sum_{k\leq n} q_k(2)$')
plt.axhline(6/16, ls='--', c='gray', label=r'$\pi(2)=3/8$')
plt.xlabel('n'); plt.ylabel('probability')
plt.title(r'Problem 1(b): $a=b=1$, $q_0=\delta_0$')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.show()
 