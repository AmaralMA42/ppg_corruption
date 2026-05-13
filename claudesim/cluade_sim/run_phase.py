"""Phase sweep — saves after each alpha slice so partial results survive timeouts."""
import numpy as np, time, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/mnt/user-data/outputs'); OUT.mkdir(exist_ok=True)
TMP = Path('/tmp')

COLORS = ['#2166ac', '#d73027', '#4dac26']   # C, D, P

def build_viz(L):
    N=L*L; idx=np.arange(N); row,col=idx//L,idx%L
    return np.stack([((row-1)%L)*L+col, row*L+(col+1)%L,
                     ((row+1)%L)*L+col, row*L+(col-1)%L], axis=1).astype(np.int32)

def payoffs(strat, viz, r, c, sig, alpha, G=5):
    isC=(strat==0).astype(float); isP=(strat==2).astype(float)
    Nc=isC+isC[viz[:,0]]+isC[viz[:,1]]+isC[viz[:,2]]+isC[viz[:,3]]
    Np=isP+isP[viz[:,0]]+isP[viz[:,1]]+isP[viz[:,2]]+isP[viz[:,3]]
    Lg=r*c*(Nc+Np-sig*Np)/G
    g0=np.arange(len(strat))
    pay=Lg[g0]+Lg[viz[:,0]]+Lg[viz[:,1]]+Lg[viz[:,2]]+Lg[viz[:,3]]
    mC=(strat==0); mP=(strat==2)
    for g in [g0, viz[:,0], viz[:,1], viz[:,2], viz[:,3]]:
        nc=Nc[g]; np_=Np[g]
        pay -= mC*c
        pay += mC*np.where(nc>0, alpha*sig*c*np_/np.where(nc>0,nc,1.0), 0.0)
        pay -= mP*c
        pay += mP*np.where(nc>0, (1-alpha)*sig*c, sig*c)
    return pay

def run(L, r, c, sig, alpha, k, G, steps, seed=None):
    if seed is not None: np.random.seed(seed)
    N=L*L; viz=build_viz(L)
    s=np.random.randint(0,3,N).astype(np.int32)
    ts=np.zeros((3,steps))
    for t in range(steps):
        pay=payoffs(s,viz,r,c,sig,alpha,G)
        ts[:,t]=np.bincount(s,minlength=3)/N
        f=np.random.randint(0,N,N); nb=viz[f,np.random.randint(0,4,N)]
        d=pay[nb]-pay[f]
        p=np.where(d>=0, 1/(1+np.exp(-d/k)), np.exp(d/k)/(1+np.exp(d/k)))
        flip=(s[f]!=s[nb])&(np.random.random(N)<p)
        s[f[flip]]=s[nb[flip]]
    return ts, s.reshape(L,L)

def pt(L, r, c, sig, alpha, k, G, steps, nsamp, eq=0.55, seed0=0):
    arr=[]
    for i in range(nsamp):
        ts,grid=run(L,r,c,sig,alpha,k,G,steps,seed=seed0+i*13)
        arr.append(ts[:,int(eq*steps):].mean(axis=1))
    arr=np.array(arr)
    return arr.mean(0), arr.std(0,ddof=1)/max(nsamp**0.5,1)

# ── params ──────────────────────────────────────────────────────
c, k, G = 1.0, 0.1, 5
L     = 50
steps = 280
nsamp = 3
eq    = 0.55

rv = np.linspace(2.5, 6.0, 11)
sv = np.linspace(0.5, 3.0, 11)
alist = [0.0, 0.2, 0.5]
na, nr, ns_ = len(alist), len(rv), len(sv)

res = np.zeros((na, 3, nr, ns_))
t0 = time.time(); done=0; total=na*nr*ns_

for ai, alpha in enumerate(alist):
    for ri, r in enumerate(rv):
        for si, sig in enumerate(sv):
            m, _ = pt(L, r, c, sig, alpha, k, G, steps, nsamp, eq)
            res[ai,:,ri,si] = m
            done += 1
            if done % 15 == 0:
                el = time.time()-t0
                eta = el/done*(total-done)
                print(f"[{done:3d}/{total}  ETA {eta:4.0f}s]  a={alpha:.1f} r={r:.2f} s={sig:.2f}"
                      f"  C={m[0]:.2f} D={m[1]:.2f} P={m[2]:.2f}", flush=True)
    # save after each alpha slice
    np.save(TMP/f'res_a{ai}.npy', res[ai])
    print(f"== alpha={alpha:.1f} slice saved ==", flush=True)

np.save(TMP/'res.npy', res)
np.save(TMP/'rv.npy', rv)
np.save(TMP/'sv.npy', sv)
print(f"Full sweep done in {time.time()-t0:.1f}s")
sys.stdout.flush()

# ── FIGURES ─────────────────────────────────────────────────────
fig, axes = plt.subplots(na, 3, figsize=(14, 4.5*na))
if na==1: axes=axes[np.newaxis,:]
cmaps=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
strat_lbl=["C (cooperator)","D (defector)","P (corrupt)"]

for ai, alpha in enumerate(alist):
    for si in range(3):
        ax=axes[ai,si]
        data=res[ai,si,:,:]
        im=ax.imshow(data, origin='lower', aspect='auto', vmin=0, vmax=1,
                     cmap=cmaps[si],
                     extent=[sv[0],sv[-1],rv[0],rv[-1]])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(f"{strat_lbl[si]},  α={alpha:.2f}", fontsize=10, fontweight='bold')
        ax.set_xlabel("σ (corruption strength)")
        ax.set_ylabel("r (synergy)")
        ax.axvline(x=1.0, color='white', lw=1.2, ls='--', alpha=0.6, label='σ=1')

fig.suptitle("Corruption Game — Steady-state fractions\nPhase diagram (σ × r)", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUT/'phase_diagram.png', dpi=150, bbox_inches='tight')
print("Saved phase_diagram.png")

# ── r-sweep line plots from phase data ──────────────────────────
sig_slice_vals = [1.05, 1.5, 2.0]
fig2, axes2 = plt.subplots(na, len(sig_slice_vals),
                            figsize=(5*len(sig_slice_vals), 4*na), sharey=True, squeeze=False)
for ai, alpha in enumerate(alist):
    for sj, sig_s in enumerate(sig_slice_vals):
        si = np.argmin(np.abs(sv - sig_s))
        ax = axes2[ai, sj]
        ax.plot(rv, res[ai,0,:,si], color=COLORS[0], lw=2, label='C')
        ax.plot(rv, res[ai,1,:,si], color=COLORS[1], lw=2, label='D')
        ax.plot(rv, res[ai,2,:,si], color=COLORS[2], lw=2, label='P')
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"α={alpha:.1f}, σ≈{sig_s:.2f}", fontsize=10)
        ax.set_xlabel("r")
        ax.grid(alpha=0.3)
        if sj==0: ax.set_ylabel("Steady-state fraction")
        if ai==0 and sj==0: ax.legend(fontsize=8)
fig2.suptitle("Strategy fractions vs synergy r — slices at fixed σ", fontsize=12)
plt.tight_layout()
fig2.savefig(OUT/'r_sweep.png', dpi=150, bbox_inches='tight')
print("Saved r_sweep.png")
