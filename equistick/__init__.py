"""
equistick: numerical search for knots whose equilateral stick number
exceeds their stick number.

Modules
-------
geometry    distances, knot-type-safe moves, Millett-Rawdon ratio
invariants  projections, PD codes, Alexander polynomial, SnapPy, HFK
certify     high-precision certificates, polishing, torus verification
torus       T(p, p+1) constructions and the symmetric no-go scan
flows       equalizer #1 (path lifting; produced the false collapse signal)
optimize    equalizers #2 (clearance floor) and #3 (safe homotopy)
reduce      stick-count reduction by annealing toward a deletable vertex
data        Eddy's stick-knot-gen data
"""
import warnings
warnings.filterwarnings('ignore')
