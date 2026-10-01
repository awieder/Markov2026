import numpy as np
import matplotlib
import matplotlib.pyplot as plt
rng=np.random.default_rng(1)
St = ['U','I','M','F','A']
p=np.array([[0,.5,.5,0,0],[.25,0,0,.5,.25],[.75,0,0,0,.25],[0,0,0,1,0],[0,0,0,0,1]])
C=np.cumsum(p,axis=1)
def run(x):
    t=0
    while x<3:
        u=rng.random(); x=int(np.searchsorted(C[x],u,side='right')); t+=1   # inverse transform
    return x,t

N=10**4
res={}
for x in range(3):
    out=[run(x) for _ in range(N)]
    final_state=np.array([o[0] for o in out]); T=np.array([o[1] for o in out])
    res[x]=(final_state,T)

exact={'U':(.5,4,4,4),'I':(5/8,2,9/5,7/3),'M':(3/8,4,5,17/5)}
print("x   h_hat  h | g_hat  g | tauF_hat tauF | tauA_hat tauA")
for x in range(3):
    f,T=res[x]; e=exact[St[x]]
    print(St[x],f"{(f==3).mean():.3f} {e[0]:.3f} | {T.mean():.3f} {e[1]:.3f} | {T[f==3].mean():.3f} {e[2]:.3f} | {T[f==4].mean():.3f} {e[3]:.3f}")

Q=p[:3,:3]; R=p[:3,3:5]
f,T=res[1]

#Plotting
fig,ax=plt.subplots(1,2,figsize=(11,4))
nmax=25
for k,(name,code) in enumerate([('folded',3),('aggregated',4)]):
    Tk=T[f==code]; ns=np.arange(1,nmax+1)
    emp=[(Tk==n).mean() for n in ns]
    hI=5/8 if code==3 else 3/8
    ex=[(np.linalg.matrix_power(Q,n-1)@R)[1,code-3]/hI for n in ns]
    ax[k].bar(ns,emp,color='C0',alpha=.6,label='simulated'); ax[k].plot(ns,ex,'ro-',ms=4,label='exact')
    ax[k].set_title(f'Start I, {name} (n={len(Tk)})'); ax[k].set_xlabel('T'); ax[k].set_ylabel('P(T=n | fate)'); ax[k].legend()
    print(name,'max |emp-exact| =',np.max(np.abs(np.array(emp)-ex)),' even-T count =',(Tk%2==0).sum())

plt.tight_layout()
plt.savefig('HW4_problem4.png')
plt.show()