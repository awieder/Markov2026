import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

a, lam_f, lam_s = 0.9, 1000.0, 10.0   # s^-1
N = 10**5

# composition sampling
U1 = rng.random(N)
B  = (U1 <= a)                        # Bernoulli(a) draw
rates = np.where(B, lam_f, lam_s)

U2 = rng.random(N)
T  = -np.log(U2) / rates              # inverse-transform draw

E_T_theory    = a/lam_f + (1-a)/lam_s
P_gt50_theory = a*np.exp(-lam_f*0.05) + (1-a)*np.exp(-lam_s*0.05)

# --- Empirical ---
E_T_emp    = T.mean()
P_gt50_emp = np.mean(T > 0.05)
print(E_T_theory, E_T_emp)
print(P_gt50_theory, P_gt50_emp)

# Plotting
def f(t):
    return a*lam_f*np.exp(-lam_f*t) + (1-a)*lam_s*np.exp(-lam_s*t)

fig, ax = plt.subplots(figsize=(7,5))
bins = np.logspace(np.log10(T.min()), np.log10(T.max()), 80)
ax.hist(T, bins=bins, density=True, alpha=0.6, color='steelblue',
        edgecolor='white', linewidth=0.3, label='Simulated dwell times')

t_grid = np.logspace(np.log10(T.min()), np.log10(T.max()), 1000)
ax.plot(t_grid, f(t_grid), 'r-', lw=2, label=r'$f(t)$')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Closed dwell time t (s)')
ax.set_ylabel('Probability density (log scale)')
ax.legend(); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('HW2_problem4.png', dpi=150)