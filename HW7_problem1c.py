import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
x = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97,
              0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87])
n = len(x)


def logg(alpha, b):
    # log of f(alpha, e^b) * e^b, up to a constant
    return n * b - np.sum(np.log(np.exp(2 * b) + (x - alpha) ** 2))


def run_chain(s, T=50_000):
    chain = np.empty((T, 2))
    z, lg, acc = np.zeros(2), logg(0.0, 0.0), 0
    noise = rng.standard_normal((T, 2))
    logu = np.log(rng.random(T))
    for t in range(T):
        y = z + s * noise[t]
        ly = logg(*y)
        if logu[t] < ly - lg:
            z, lg, acc = y, ly, acc + 1
        chain[t] = z
    return chain, acc / T


def iact(v):
    # integrated autocorrelation time, sum truncated with Sokal's window (c = 5)
    v = v - v.mean()
    N = len(v)
    f = np.fft.rfft(v, 2 * N)
    acf = np.fft.irfft(f * np.conj(f))[:N]
    acf /= acf[0]
    taus = 2 * np.cumsum(acf) - 1
    ok = np.arange(N) >= 5 * taus
    return taus[np.argmax(ok)]


def wquantile(vals, w, q):
    i = np.argsort(vals)
    c = np.cumsum(w[i])
    return np.interp(q, c / c[-1], vals[i])


# three step sizes: acceptance rate, IACT of alpha, trace of first 400 values
steps = [0.02, 0.5, 5.0]
chains, taus = {}, {}
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
print(f"{'s':>5} {'acceptance':>11} {'IACT(alpha)':>12}")
for ax, s in zip(axes, steps):
    chains[s], acc = run_chain(s)
    taus[s] = iact(chains[s][:, 0])
    print(f"{s:5.2f} {acc:11.3f} {taus[s]:12.1f}")
    ax.plot(chains[s][:400, 0])
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(f"s = {s}")
axes[-1].set_xlabel("step")
fig.tight_layout()
fig.savefig("HW7_problem1c_traces.png", dpi=150)

# exact posterior on a grid in (alpha, b)
a_grid = np.linspace(-1, 4, 600)
b_grid = np.linspace(-2.5, 1.5, 600)
A, B = np.meshgrid(a_grid, b_grid, indexing="ij")
L = n * B - np.sum(np.log(np.exp(2 * B)[..., None] + (x - A[..., None]) ** 2), axis=-1)
P = np.exp(L - L.max())
P /= P.sum()
w_a, w_b = P.sum(axis=1), P.sum(axis=0)
beta_grid = np.exp(b_grid)

# best s: discard first 5000 steps, contours + samples
best = min(steps, key=taus.get)
samp = chains[best][5000:]
al, be = samp[:, 0], np.exp(samp[:, 1])
fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(A, B, P, levels=8, colors="k", linewidths=0.8)
ax.scatter(samp[::10, 0], samp[::10, 1], s=3, alpha=0.3)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\log\beta$")
ax.set_title(f"Samples (s = {best}) over exact posterior contours")
fig.tight_layout()
fig.savefig("HW7_problem1c_contours.png", dpi=150)

print(f"\nbest s = {best}")
print("posterior mean and 90% interval (5th to 95th percentile)")
print(f"alpha chain: {al.mean():.3f}  [{np.percentile(al, 5):.3f}, {np.percentile(al, 95):.3f}]")
print(f"alpha grid : {a_grid @ w_a:.3f}  [{wquantile(a_grid, w_a, .05):.3f}, {wquantile(a_grid, w_a, .95):.3f}]")
print(f"beta  chain: {be.mean():.3f}  [{np.percentile(be, 5):.3f}, {np.percentile(be, 95):.3f}]")
print(f"beta  grid : {beta_grid @ w_b:.3f}  [{wquantile(beta_grid, w_b, .05):.3f}, {wquantile(beta_grid, w_b, .95):.3f}]")
print(f"sample mean of x: {x.mean():.3f}   sample median of x: {np.median(x):.3f}")