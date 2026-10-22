import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(5)
lam, H = 4.0, 200_000


def hour_by_gaps():
    # exponential gaps until the hour is used up
    pts, t = [], rng.exponential(1 / lam)
    while t < 1:
        pts.append(t)
        t += rng.exponential(1 / lam)
    return np.array(pts)


def hour_by_count():
    # Poisson count, then uniform points
    return np.sort(rng.random(rng.poisson(lam)))


hours = {"exponential gaps": [hour_by_gaps() for _ in range(H)],
         "Poisson count + uniforms": [hour_by_count() for _ in range(H)]}

# histogram of N(1) vs Poisson(4), mean and variance
k = np.arange(16)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, hrs) in zip(axes, hours.items()):
    N = np.array([len(h) for h in hrs])
    print(f"{name:26s} mean N(1) = {N.mean():.4f}  var N(1) = {N.var():.4f}")
    ax.bar(k, np.bincount(N, minlength=16)[:16] / H, alpha=0.6, label="simulated")
    ax.plot(k, stats.poisson.pmf(k, lam), "ro-", ms=4, label="Poisson(4) pmf")
    ax.set_title(name)
    ax.set_xlabel("N(1)")
    ax.legend()
fig.tight_layout()
fig.savefig("HW7_problem3d_counts.png", dpi=150)

# rasters: ten realizations of each construction
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, (name, hrs) in zip(axes, hours.items()):
    for r in range(10):
        ax.vlines(hrs[r], r + 0.1, r + 0.9)
    ax.set_title(name)
    ax.set_xlabel("time in hour")
    ax.set_xlim(0, 1)
axes[0].set_ylabel("realization")
fig.tight_layout()
fig.savefig("HW7_problem3d_rasters.png", dpi=150)

# T3 against its Gamma(3, 4) density
T3 = rng.exponential(1 / lam, size=(H, 3)).sum(axis=1)
tt = np.linspace(0, 3, 400)
plt.figure(figsize=(6.5, 4.2))
plt.hist(T3, bins=100, range=(0, 3), density=True, alpha=0.5, label="simulated $T_3$")
plt.plot(tt, stats.gamma.pdf(tt, a=3, scale=1 / lam), "r", label="Gamma(3, 4) density")
plt.xlabel("t")
plt.legend()
plt.tight_layout()
plt.savefig("HW7_problem3d_T3.png", dpi=150)

# given N(1) = 3, and correlation of half-hour counts over all hours
for name, hrs in hours.items():
    three = [h for h in hrs if len(h) == 3]
    p0 = np.mean([np.sum(h < 0.5) == 0 for h in three])
    ET1 = np.mean([h[0] for h in three])
    first = np.array([np.sum(h < 0.5) for h in hrs])
    second = np.array([len(h) for h in hrs]) - first
    print(f"{name:26s} P(N(1/2)=0 | N(1)=3) = {p0:.4f}  E[T1 | N(1)=3] = {ET1:.4f}  "
          f"corr(first, second half) = {np.corrcoef(first, second)[0, 1]:+.4f}")

# total variation distance Binomial(t/dt, lam dt) vs Poisson(lam t), t = 1
dts = 1 / np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
kk = np.arange(80)
tv = np.array([0.5 * np.abs(stats.binom.pmf(kk, round(1 / d), lam * d) - stats.poisson.pmf(kk, lam)).sum()
               for d in dts])
slope, icpt = np.polyfit(np.log(dts), np.log(tv), 1)
print(f"\nlog-log slope of TV against dt: {slope:.3f}")
plt.figure(figsize=(6, 4.5))
plt.loglog(dts, tv, "o", label="TV distance")
plt.loglog(dts, np.exp(icpt) * dts ** slope, "r-", label=f"fit, slope = {slope:.2f}")
plt.xlabel(r"$\Delta t$")
plt.ylabel("TV(Binomial, Poisson)")
plt.legend()
plt.tight_layout()
plt.savefig("HW7_problem3d_tv.png", dpi=150)