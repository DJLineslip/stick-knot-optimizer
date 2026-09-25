import numpy as _np
_np.float=float; _np.int=int; _np.complex=complex
import numpy as np, math, sys, warnings, time
warnings.filterwarnings("ignore")
import io, contextlib
import spherogram
from pyknotid.spacecurves import Knot
from sym import polygon, min_nonadj, rand_rot

def pd_list(P):
    R=rand_rot(); Q=P@R.T; Q=np.vstack([Q,Q[:1]])
    with contextlib.redirect_stdout(io.StringIO()):
        k=Knot(Q, add_closure=False, verbose=False)
        s=k.planar_diagram()
    # parse "PD with n: X_{a,b,c,d} ..."
    if ':' not in s: return []
    body=s.split(':',1)[1].strip()
    out=[]
    for tok in body.split():
        tok=tok.strip()
        if not tok.startswith('X_{'): continue
        nums=tok[3:-1].split(',')
        out.append(tuple(int(x) for x in nums))
    return out

def alex(P):
    pd=pd_list(P)
    if len(pd)==0: return '1', 0
    L=spherogram.Link(pd)
    L.simplify('global')
    if len(L.crossings)==0: return '1', 0
    a=L.alexander_polynomial()
    return str(a), len(L.crossings)

def torus_alex(p,q):
    import sympy as sp
    t=sp.symbols('t')
    e=sp.cancel((t**(p*q)-1)*(t-1)/((t**p-1)*(t**q-1)))
    return sp.Poly(sp.expand(e),t)

if __name__=="__main__":
    m=int(sys.argv[1])
    results={}
    rs=[x for x in np.linspace(-3,3,61) if abs(x)>1e-9 and abs(abs(x)-1)>1e-9]
    Hs=np.linspace(0.05,2.0,14)
    for k in range(1,2*m):
        seen={}
        for r in rs:
            for H in Hs:
                P=polygon(m,k,r,H)
                mu=min_nonadj(P)
                edge=np.linalg.norm(P[1]-P[0])
                if mu<1e-6*edge: continue
                try:
                    a,c=alex(P)
                except Exception as ex:
                    continue
                if a not in seen: seen[a]=(round(r,2),round(H,2),c,mu/edge)
        results[k]=seen
        print(f"m={m} k={k}: ", {a:v for a,v in seen.items()}, flush=True)
