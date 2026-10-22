import numpy as np

rng = np.random.default_rng(11)
alpha, beta, R = 1.5, 0.8, 2000


def scatter(est):
    q25, q75 = np.percentile(est, [25, 75])
    return (q75 - q25) / 1.349


print(f"{'n':>6} {'mean':>8} {'median':>8}   (IQR/1.349 over {R} data sets)")
for n in (10, 100, 1000):
    data = alpha + beta * np.tan(rng.uniform(-np.pi / 2, np.pi / 2, size=(R, n)))
    print(f"{n:6d} {scatter(data.mean(axis=1)):8.3f} {scatter(np.median(data, axis=1)):8.3f}")