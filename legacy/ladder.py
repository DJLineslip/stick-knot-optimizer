import numpy as np, math, subprocess
def load(name):
    txt=subprocess.run(['curl','-s',f'https://raw.githubusercontent.com/thomaseddy/stick-knot-gen/master/stick_number/mseq_knots/{name}.txt'],capture_output=True,text=True).stdout
    return np.array([[float(x) for x in l.split()] for l in txt.strip().splitlines()])
# (knot, bridge index) ; bridge-tight means n = 2b+2
for name,b in [('3_1',2),('8_19',3),('8_20',3),('5_1',2),('5_2',2),('10_124',3)]:
    P=load(name); n=len(P)
    E=[np.linalg.norm(P[(i+1)%n]-P[i]) for i in range(n)]
    beta=[]
    for i in range(n):
        a=P[i-1]-P[i]; c=P[(i+1)%n]-P[i]
        beta.append(math.acos(np.clip(a@c/np.linalg.norm(a)/np.linalg.norm(c),-1,1)))
    S=sum(beta)
    Lchains=sum(np.linalg.norm(P[(i+1)%n]-P[i-1]) for i in range(n))
    bound=n*math.pi-2*math.pi*b
    print(f"{name}: n={n}, b={b}, edge lengths in [{min(E):.6f},{max(E):.6f}], sum interior angles={S:.3f} (bound nπ-2πb={bound:.3f}), total 2nd-neighbour chain length={Lchains:.3f}, bridge-tight={n==2*b+2}")
