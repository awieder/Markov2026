import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
lam = np.array([3, 2, 1.5, 1.2, 1, 0.75])
n = len(lam)


def recursion(lam):
    n = len(lam)
    EM = np.zeros(1 << n)
    last = np.zeros((1 << n, n))
    for S in range(1, 1 << n):
        mem = [i for i in range(n) if S >> i & 1]
        L = sum(lam[i] for i in mem)
        EM[S] = 1 / L
        if len(mem) == 1:
            last[S, mem[0]] = 1
        for i in mem:
            rest = S & ~(1 << i)
            EM[S] += lam[i] / L * EM[rest]
            if len(mem) > 1:
                last[S] += lam[i] / L * last[rest]
    return EM[-1], last[-1]


def alternating(lam):
    return sum((-1) ** (k + 1) / sum(lam[i] for i in T)
               for k in range(1, len(lam) + 1) for T in itertools.combinations(range(len(lam)), k))


EM, p_last = recursion(lam)
print(f"E[M] recursion   : {EM:.5f}")
print(f"E[M] alternating : {alternating(lam):.5f}")

N = 200_000
X = rng.exponential(1 / lam, size=(N, n))
M = X.max(axis=1)
sim_last = np.bincount(X.argmax(axis=1), minlength=n) / N
print(f"E[M] simulated   : {M.mean():.5f}")
print("student  P(last) exact  simulated")
for i in range(n):
    print(f"{i + 1:5d}    {p_last[i]:.4f}        {sim_last[i]:.4f}")

# histogram of M against the exact density d/dt prod(1 - e^{-lam_i t})
t = np.linspace(0, M.max(), 600)
dens = sum(lam[i] * np.exp(-lam[i] * t) * np.prod([1 - np.exp(-lam[j] * t) for j in range(n) if j != i], axis=0)
           for i in range(n))
plt.figure(figsize=(7, 4.5))
plt.hist(M, bins=120, density=True, alpha=0.5, label="simulated M")
plt.plot(t, dens, "r", label="exact density")
plt.xlabel("t (hours)")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("HW7_problem2d_hist.png", dpi=150)

# first finishing time does not depend on who finishes first
mn, who = X.min(axis=1), X.argmin(axis=1)
print(f"\nmean of first finishing time given who is first (1/Lambda = {1 / lam.sum():.4f}):")
print(" ".join(f"{mn[who == i].mean():.4f}" for i in range(n)))

# rates lambda_i = i, k = 1..12
ks = np.arange(1, 13)
EMk = [recursion(np.arange(1, k + 1, dtype=float))[0] for k in ks]
plt.figure(figsize=(6.5, 4.2))
plt.plot(ks, EMk, "o-")
plt.xlabel("k (number of students)")
plt.ylabel("E[M]")
plt.tight_layout()
plt.savefig("HW7_problem2d_EM_vs_k.png", dpi=150)
print("E[M] for k = 1..12:", " ".join(f"{e:.4f}" for e in EMk))