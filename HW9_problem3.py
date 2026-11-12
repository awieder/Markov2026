import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import brentq


Q=np.array([[-2,2,0,0],[1,-3,2,0],[0,2,-4,2],[0,0,3,-3]],float)
pi=np.array([3,6,6,4])/19
x0=np.array([1,0,0,0.])
lam2=sorted(np.linalg.eigvals(Q).real)[-2]
t=np.linspace(0,5,501)

X=np.array([x0@expm(s*Q) for s in t])
TV=0.5*np.abs(X-pi).sum(1)

f=lambda s:0.5*np.abs(x0@expm(s*Q)-pi).sum()-0.01

t01=brentq(f,0.1,10)
print("lam2",lam2,"t(TV=.01)",t01,"ln100/|l2|",np.log(100)/abs(lam2))


# asymptotic constant: TV ~ C e^{lam2 t}
C=TV[-1]/np.exp(lam2*5); print("C",C, "t from C:",np.log(C/0.01)/abs(lam2))
cols=["#2a78d6","#eb6834","#1baf7a","#eda100"]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for i in range(4): ax[0].plot(t,X[:,i],color=cols[i],lw=2,label=f"state {i}")
for i in range(4): ax[0].axhline(pi[i],color=cols[i],lw=.8,ls=":")
ax[0].set(xlabel="t (hours)",ylabel="P(X_t = i)",title="x_i(t), starting empty (dotted = π_i)")
ax[0].legend(frameon=False)
ax[1].semilogy(t,TV,color="#2a78d6",lw=2,label="TV(t)")
ax[1].semilogy(t,C*np.exp(lam2*t),color="#52514e",ls="--",lw=1.5,label=f"C·e^{{λ₂t}}, λ₂={lam2:.3f}")
ax[1].axhline(.01,color="#eb6834",lw=1); ax[1].axvline(t01,color="#eb6834",lw=1)
ax[1].set(xlabel="t (hours)",ylabel="TV distance (log)",title=f"TV = 0.01 at t ≈ {t01:.2f} h",ylim=(1e-4,1))
ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top","right"]].set_visible(False); a.grid(alpha=.2)
plt.tight_layout(); plt.savefig("HW9_problem3.png",dpi=130)