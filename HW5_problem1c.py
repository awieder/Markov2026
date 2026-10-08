import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
a, b = 0.3, 0.1
p = np.zeros((5, 5))
for k in range(5):
    if k < 4: p[k, k+1] = a*(4-k)/4
    if k > 0: p[k, k-1] = b*k/4
    p[k, k] = 1 - p[k].sum()
assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()

A = (p - np.eye(5)).T.copy()
A[-1, :] = 1
right = np.zeros(5); right[-1] = 1
pi = np.linalg.solve(A, right)

theta = a/(a+b)
binom = np.array([comb(4, k)*theta**k*(1-theta)**(4-k) for k in range(5)])
print("pi (solve) =", pi)
print("binomial   =", binom)
print("max diff   =", np.abs(pi-binom).max())
 
# iterate q_n
q = np.zeros(5); q[0] = 1
Q = [q.copy()]
n_hit = None
for n in range(1, 400):
    q = q @ p
    Q.append(q.copy())
    if n_hit is None and np.abs(q-pi).max() < 1e-6:
        n_hit = n
Q = np.array(Q)
print("smallest n with max err < 1e-6:", n_hit)
 
ev = np.linalg.eigvals(p)
mods = np.sort(np.abs(ev))[::-1]
print("eigenvalues:", np.round(np.sort(ev.real)[::-1], 5))
print("SLEM =", mods[1])
print("predicted n: ~", np.log(1e-6)/np.log(mods[1]))
#Plotting
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
w = 0.38; k = np.arange(5)
ax[0].bar(k-w/2, pi, w, label=r'$\pi$ (linear solve)')
ax[0].bar(k+w/2, binom, w, label=r'Binomial$(4,\theta)$')
ax[0].set_xlabel('k'); ax[0].set_ylabel('probability'); ax[0].legend()
ax[0].set_title(r'Stationary $\pi$, $a=0.3,\ b=0.1$')
for j in range(5):
    ax[1].plot(Q[:61, j], label=f'$q_n({j})$')
    ax[1].axhline(pi[j], ls=':', c='gray', lw=.8)
# ax[1].axvline(n_hit, ls='--', c='k', lw=.8)
ax[1].set_xlabel('n'); ax[1].set_ylabel('probability'); ax[1].legend(fontsize=8)
ax[1].set_title(r'$q_n$ from $\delta_0$ (dotted: $\pi$)')
plt.tight_layout(); plt.savefig("HW5_problem1c.png", dpi=150)