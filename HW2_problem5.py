import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)
E_z_theory   = 1/3
Var_z_theory = 1/18
SD_z_theory  = np.sqrt(Var_z_theory)
ratio_theory = SD_z_theory / E_z_theory

# Part b
Nb = 10**5
U = rng.random(Nb)
Z = 1 - np.sqrt(U)

print("\n=== Part (b) inversion sampling, N=10^5 ===")
print(f"Empirical mean  = {Z.mean():.6f}  (theory 1/3 = {1/3:.6f})")

# # Part c
R = 10**3          # realizations
N_target = 10**4    # grow up to N = 10^4
N0 = 3

C = np.ones(R, dtype=np.int64)   # C(3) = 1 for every realization

for N in range(N0, N_target):   
    p = C / N                     # length-R vector of probabilities, mu=1
    draws = rng.random(R) < p     # length-R vector of Bernoulli draws
    C += draws                    # vectorized update across all R realizations

z_final = C / N_target

emp_mean_c = z_final.mean()
emp_sd_c   = z_final.std()
ratio_c    = emp_sd_c / emp_mean_c

print("\n=== Part (c) simulation, N=10^4, R=10^3 ===")
print(f"Empirical mean(z)   = {emp_mean_c:.6f}  (theory {E_z_theory:.6f})")
print(f"Empirical SD/mean   = {ratio_c:.6f}  (theory {ratio_theory:.6f})")
print(f"Smallest core seen  = {C.min()}  (z = {C.min()/N_target:.4f})")
print(f"Largest core seen   = {C.max()}  (z = {C.max()/N_target:.4f})")

# Plotting
def h(z):
    return 2*(1-z)

fig, ax = plt.subplots(figsize=(7,5))
bins = np.linspace(0, 1, 40)
ax.hist(z_final, bins=bins, density=True, alpha=0.65, color='seagreen',
        edgecolor='white', linewidth=0.4, label=f'Simulated $z=C/N$ ($R={R}$ runs)')

zz = np.linspace(0, 1, 300)
ax.plot(zz, h(zz), 'r-', lw=2, label=r'$h(z)=2(1-z)$')

ax.set_xlabel('z = C(N)/N')
ax.set_ylabel('Probability density')
ax.set_title(r'Core fraction $z=C/N$ at $N=10^4$, $\mu=1$')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('HW2_problem5.png', dpi=150)
print("\nPlot saved.")