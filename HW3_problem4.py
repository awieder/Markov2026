import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


rng = np.random.default_rng(0)

d0 = 10
R = 2 * 10**4
T = 10**4


def run_sim(n_lions):
    # D[k] = separation between lion k and the lamb
    D = np.full((n_lions, R), d0, dtype=np.int64)
    alive = np.ones(R, dtype=bool)

    S = np.zeros(T + 1)
    S[0] = 1.0

    for t in range(1, T + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            # everyone's dead, rest stays 0
            break

        # lamb takes one step, shared by all lions this turn
        lamb_step = rng.choice([-1, 1], size=len(idx))

        caught = np.zeros(len(idx), dtype=bool)
        for k in range(n_lions):
            lion_step = rng.choice([-1, 1], size=len(idx))
            D[k, idx] += lion_step - lamb_step
            caught = caught | (D[k, idx] == 0)

        alive[idx[caught]] = False
        S[t] = alive.mean()

    return S


S1 = run_sim(1)
S2 = run_sim(2)

t = np.arange(T + 1)

# power law fit, S(t) ~ t^-beta, over 100 <= t <= 10000
# have to drop any t where S is 0 
lo, hi = 100, 10000
window = (t >= lo) & (t <= hi)

mask1 = window & (S1 > 0)
slope1, b1 = np.polyfit(np.log10(t[mask1]), np.log10(S1[mask1]), 1)
beta1 = -slope1

mask2 = window & (S2 > 0)
slope2, b2 = np.polyfit(np.log10(t[mask2]), np.log10(S2[mask2]), 1)
beta2 = -slope2

print("beta1 =", round(beta1, 3))
print("beta2 =", round(beta2, 3))

# how many points got dropped from the fit because S hit 0
print("dropped points (N=1):", window.sum() - mask1.sum())
print("dropped points (N=2):", window.sum() - mask2.sum())

# theory curve for the single-lion case, no fitting involved
tt = np.arange(1, T + 1)
S1_theory = erf(d0 / (2 * np.sqrt(tt)))

# little table asked for in the problem
print()
print("t\tS2(t)\tS1(t)^2")
for tval in [100, 1000, 10000]:
    print(f"{tval}\t{S2[tval]:.5f}\t{S1[tval]**2:.5f}")

# plot everything together
plt.figure(figsize=(7.5, 5.5))

m1 = S1 > 0
m2 = S2 > 0
plt.loglog(t[m1], S1[m1], '.', ms=3, label='S1(t), N=1')
plt.loglog(t[m2], S2[m2], '.', ms=3, label='S2(t), N=2')
plt.loglog(t[m1], S1[m1]**2, '.', ms=3, label='S1(t)^2')
plt.loglog(tt, S1_theory, label='erf(d0 / 2 sqrt(t))')

plt.xlabel('t')
plt.ylabel('survival probability')
plt.title(f'lamb survival vs lions, beta1={beta1:.3f}, beta2={beta2:.3f}\n'
          f'(fit range {lo}-{hi}, R={R}, T={T})')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('HW3_problem4.png', dpi=150)