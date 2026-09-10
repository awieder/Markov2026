import numpy as np
import matplotlib.pyplot as plt
import time


def f_(x):
    #Gamma(2,1) density
    return x * np.exp(-x)


def g_(x, lam):
    #Exponential(lam) density
    return lam * np.exp(-lam * x)


def rejection_sample(N, lam, c, batch=5000, seed=None):
    rng = np.random.default_rng(seed)
    accepted = []
    total_proposals = 0
    t0 = time.perf_counter()
 
    while len(accepted) < N:
        U = rng.random(batch)
        X = -np.log(1 - U) / lam          
        # inverse-CDF draw from Exp(lam)
        U2 = rng.random(batch)
 
        f_x = f_(X)
        g_x = g_(X, lam)
        mask = U2 < f_x / (c * g_x)        
        # acceptance test
 
        idx_accept = np.nonzero(mask)[0]
        needed = N - len(accepted)
 
        if len(idx_accept) <= needed:
            accepted.extend(X[idx_accept])
            total_proposals += batch
        else:
            cutoff = idx_accept[needed - 1]     
            # position of the N-th acceptance in this batch
            accepted.extend(X[idx_accept[:needed]])
            total_proposals += cutoff + 1       
            # only count proposals actually needed
 
    elapsed = time.perf_counter() - t0
    return np.array(accepted), total_proposals, elapsed



def run_case(lam, c, N=10_000, seed=0):
    samples, total_proposals, elapsed = rejection_sample(N, lam, c, seed=seed)
    accept_frac = N / total_proposals
    theoretical = 1 / c
    mean_time_per_sample = elapsed / N
    return {
        "lam": lam, "c": c, "N": N,
        "samples": samples,
        "total_proposals": total_proposals,
        "elapsed": elapsed,
        "emp_accept_frac": accept_frac,
        "theoretical_accept_frac": theoretical,
        "mean_time_per_sample": mean_time_per_sample,
    }

# Case 1: lambda = 1/2, c = 4/e
lam1 = 0.5
c1 = 4 / np.e
res1 = run_case(lam1, c1, N=10_000, seed=1)
 
# Case 2: lambda = 0.2, c = 1/(0.16 e) 
lam2 = 0.2
c2 = 1 / (0.16 * np.e)
res2 = run_case(lam2, c2, N=10_000, seed=2)
 
for res in (res1, res2):
    print(f"--- lambda = {res['lam']} ---")
    print(f"  c (envelope constant)       : {res['c']:.4f}")
    print(f"  N accepted                  : {res['N']}")
    print(f"  total proposals             : {res['total_proposals']}")
    print(f"  empirical acceptance frac   : {res['emp_accept_frac']:.4f}")
    print(f"  theoretical acceptance 1/c  : {res['theoretical_accept_frac']:.4f}")
    print(f"  elapsed time (s)            : {res['elapsed']:.4f}")
    print(f"  mean time per accepted (µs) : {res['mean_time_per_sample']*1e6:.3f}")
    print()
 
# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_grid = np.linspace(0, 15, 500)
 
for ax, res in zip(axes, (res1, res2)):
    ax.hist(res["samples"], bins=60, density=True, alpha=0.6,
            color="#4C72B0", edgecolor="white", label="Empirical (accepted samples)")
    ax.plot(x_grid, f_(x_grid), color="#C44E52", lw=2, label=r"$f(x)=xe^{-x}$")
    ax.set_title(f"$\\lambda={res['lam']}$,  $c={res['c']:.4f}$\n"
                 f"empirical accept={res['emp_accept_frac']:.3f}, "
                 f"theoretical $1/c$={res['theoretical_accept_frac']:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_xlim(0, 15)
    ax.legend()
 
plt.tight_layout()
plt.savefig("HW2_problem3.png", dpi=300)
plt.show()