"""Time series + spatial snapshots at key parameter points."""
import numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/mnt/user-data/outputs'); OUT.mkdir(exist_ok=True)
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

c, k, G = 1.0, 0.1, 5

# Interesting regimes based on paper notes + phase diagram intuition
cases = [
    # (r, sigma, alpha, label)
    (3.5, 1.05, 0.20, "Cyclic oscillations\n(r=3.5, σ=1.05, α=0.2)"),
    (3.5, 2.00, 0.00, "P dominates — no bribery\n(r=3.5, σ=2.0, α=0)"),
    (3.5, 2.00, 0.30, "Coexistence with bribery\n(r=3.5, σ=2.0, α=0.3)"),
    (4.5, 1.50, 0.50, "High bribery, C recovers\n(r=4.5, σ=1.5, α=0.5)"),
    (5.5, 1.20, 0.20, "High r, C+P coexist\n(r=5.5, σ=1.2, α=0.2)"),
    (2.8, 1.50, 0.00, "Low r, P barely beats D\n(r=2.8, σ=1.5, α=0)"),
]

n = len(cases)
nsamp = 5
steps = 700
L_ts  = 60
L_sn  = 80

print(f"Running {n} cases x {nsamp} samples x {steps} steps ...")
t0 = time.time()

all_ts_data = []
snap_data   = []

for ci, (r, sig, alpha, lbl) in enumerate(cases):
    print(f"  [{ci+1}/{n}] {lbl.split(chr(10))[0]}", flush=True)
    samples = []
    for s in range(nsamp):
        ts, _ = run(L_ts, r, c, sig, alpha, k, G, steps, seed=100+ci*31+s*7)
        samples.append(ts)
    all_ts_data.append(np.array(samples))   # (nsamp, 3, steps)
    # snapshot at longer run
    _, grid = run(L_sn, r, c, sig, alpha, k, G, steps, seed=999+ci)
    snap_data.append(grid)

print(f"Simulations done in {time.time()-t0:.1f}s")

# ── TIME SERIES FIGURE ──────────────────────────────────────────
cols = 2; rows = (n+1)//2
fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5*rows), squeeze=False)
axes = axes.flatten()
strat_names = ['C', 'D', 'P']

for ci, (r, sig, alpha, lbl) in enumerate(cases):
    ax = axes[ci]
    ts_arr = all_ts_data[ci]   # (nsamp, 3, steps)
    mean_ts = ts_arr.mean(axis=0)
    for samp in ts_arr:
        for si, cl in enumerate(COLORS):
            ax.plot(samp[si], color=cl, alpha=0.12, lw=0.7)
    for si, (nm, cl) in enumerate(zip(strat_names, COLORS)):
        ax.plot(mean_ts[si], color=cl, lw=2.2, label=nm)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(lbl, fontsize=9, fontweight='bold')
    ax.set_xlabel("MCS"); ax.set_ylabel("ρ")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)
    # shade equilibration region
    eq_t = int(0.4*steps)
    ax.axvspan(0, eq_t, alpha=0.06, color='gray')

for ci in range(n, len(axes)): axes[ci].set_visible(False)
fig.suptitle("Corruption Game — Time evolution at selected regimes\n"
             "(shaded = equilibration period; lines = 5 samples + mean)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT/'timeseries.png', dpi=150, bbox_inches='tight')
print("Saved timeseries.png")

# ── SPATIAL SNAPSHOTS ────────────────────────────────────────────
cmap_snap = plt.matplotlib.colors.ListedColormap(COLORS)
cols_sn = 3; rows_sn = (n+2)//3
fig2, axes2 = plt.subplots(rows_sn, cols_sn, figsize=(4.8*cols_sn, 4.8*rows_sn), squeeze=False)
axes2f = axes2.flatten()

for ci, (r, sig, alpha, lbl) in enumerate(cases):
    ax = axes2f[ci]
    im = ax.imshow(snap_data[ci], vmin=0, vmax=2, cmap=cmap_snap, interpolation='none')
    ax.set_title(lbl, fontsize=8.5, fontweight='bold')
    ax.axis('off')

for ci in range(n, len(axes2f)): axes2f[ci].set_visible(False)

# shared colorbar
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for c,l in zip(COLORS,['C','D','P'])]
fig2.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
            bbox_to_anchor=(0.5, -0.02))
fig2.suptitle(f"Spatial configurations at t={steps} MCS  (L={L_sn}×{L_sn})", fontsize=12)
plt.tight_layout()
fig2.savefig(OUT/'snapshots.png', dpi=150, bbox_inches='tight')
print("Saved snapshots.png")
print(f"Total: {time.time()-t0:.1f}s")
