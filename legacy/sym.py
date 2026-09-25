import numpy as _np
_np.float = float
_np.int = int
_np.complex = complex
import numpy as np, itertools, math, warnings
warnings.filterwarnings("ignore")
import spherogram
from pyknotid.spacecurves import Knot

def polygon(m, k, r, H):
    # 2m vertices; vertex j at angle j*pi*k/m, radius 1 (even) or r (odd, signed), height +H/-H
    n = 2*m
    pts = []
    for j in range(n):
        ang = j*math.pi*k/m
        rho = 1.0 if j % 2 == 0 else r
        z = H if j % 2 == 0 else -H
        pts.append([rho*math.cos(ang), rho*math.sin(ang), z])
    return np.array(pts)

def seg_dist(p1,p2,q1,q2):
    # min distance between segments p1p2 and q1q2 (sampling-free closed form via clamped params)
    d1=p2-p1; d2=q2-q1; r=p1-q1
    a=d1@d1; e=d2@d2; f=d2@r
    c=d1@r; b=d1@d2; den=a*e-b*b
    cands=[]
    def clamp(x): return min(1,max(0,x))
    if den>1e-14:
        s=clamp((b*f-c*e)/den)
    else:
        s=0.0
    t=(b*s+f)/e
    if t<0: t=0; s=clamp(-c/a)
    elif t>1: t=1; s=clamp((b-c)/a)
    best=np.linalg.norm(p1+s*d1-(q1+t*d2))
    # also check endpoints for robustness
    for (s0,t0) in [(0,None),(1,None)]:
        pt=p1+s0*d1; tt=clamp(((pt-q1)@d2)/e); best=min(best,np.linalg.norm(pt-(q1+tt*d2)))
    for t0 in [0,1]:
        qt=q1+t0*d2; ss=clamp(((qt-p1)@d1)/a); best=min(best,np.linalg.norm(p1+ss*d1-qt))
    return best

def min_nonadj(P):
    n=len(P); best=1e9
    for i in range(n):
        for j in range(i+2,n):
            if i==0 and j==n-1: continue
            best=min(best,seg_dist(P[i],P[(i+1)%n],P[j],P[(j+1)%n]))
    return best

rng=np.random.default_rng(1)
def rand_rot():
    q=rng.normal(size=4); q/=np.linalg.norm(q)
    a,b,c,d=q
    return np.array([[a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c)],
                     [2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b)],
                     [2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d]])

def jones_of_polygon(P):
    R=rand_rot()
    Q=(P@R.T)
    Q=np.vstack([Q,Q[:1]])  # close
    k=Knot(Q, add_closure=False, verbose=False)
    pd=k.planar_diagram()
    return pd

if __name__=="__main__":
    P=polygon(3,2,0.3,0.5)
    print(P.round(3))
    print(min_nonadj(P))
    print(jones_of_polygon(P))
