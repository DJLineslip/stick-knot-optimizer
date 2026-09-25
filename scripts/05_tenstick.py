"""
05_tenstick.py: equilateral 10-stick versions of the 4-bridge knots with
stick number exactly 10.

Pipeline per knot: take Eddy's equilateral 11- or 12-stick polygon ->
anneal down to 10 sticks (equistick.reduce), check SnapPy identifies the
right knot -> fatten -> safe homotopy to equal lengths (equalizer #3) ->
Millett-Rawdon certificate in 40 digits -> SnapPy identification again.
Results are appended to results/tenstick.log and coordinates saved.

usage: python 05_tenstick.py K11n71,K11n75,... [seconds_per_knot]
"""
import sys, time
import numpy as np
from equistick.data import load_eddy, eddy_available, seed_for, TEN_STICK_19, exact_stick_numbers
from equistick.geometry import normalize, mr_ratio
from equistick.invariants import identify
from equistick.reduce import reduce_to
from equistick.optimize import fatten, homotopy_equalize
from equistick.certify import mr_certificate_mp

class CandidateRejected(ValueError):
    """A computed check rejected this candidate, not an execution error."""


def validate_candidate(V, name, target):
    """Check the exact supplied coordinates, not an earlier optimizer iterate.

    TEN_STICK_19 cites Cantarella et al., JKTR 2026, for s(K) = 10.
    Certificates and SnapPy identifications are numerical, not formal proofs.
    """
    if V.shape != (target, 3) or not np.isfinite(V).all():
        raise CandidateRejected('invalid coordinate shape or nonfinite coordinates')
    known = 10 if name in TEN_STICK_19 else exact_stick_numbers().get(name)
    if known != target:
        raise CandidateRejected('target is not the known exact stick number')
    if not mr_ratio(V)[0] < 1:
        raise CandidateRejected('Millett and Rawdon ratio is not below 1')
    certificate = mr_certificate_mp(V)
    if not bool(certificate[3]):
        raise CandidateRejected('high precision certificate failed')
    if any(identify(V, seed=seed) != name for seed in (1, 2, 3)):
        raise CandidateRejected('final projection identification mismatch')
    return certificate


def run(name, budget=240, target=10, out='../results'):
    if not eddy_available(name):
        return 'no starting data in Eddy repository'
    rng = np.random.default_rng(seed_for(name))
    V0 = load_eddy(name)
    t0, nreal, attempts = time.monotonic(), 0, 0
    print(f'{name}: seed={seed_for(name)} budget={budget:g}s target={target} start_vertices={len(V0)}', flush=True)
    while time.monotonic() - t0 < budget:
        attempts += 1
        print(f'{time.monotonic() - t0:.3f}s reduction attempt {attempts}', flush=True)
        V = reduce_to(V0.copy(), target, rng)
        if V is None or identify(V) != name:
            continue
        print(f'{time.monotonic() - t0:.3f}s reduction identified, fattening', flush=True)
        nreal += 1
        Vf = fatten(V, rng, 400)
        for floor in (0.9, 0.5, 0.2):
            print(f'{time.monotonic() - t0:.3f}s homotopy floor={floor}', flush=True)
            E, t, mu0, done = homotopy_equalize(Vf, mu_floor=floor, tlimit=120)
            print(f'{time.monotonic() - t0:.3f}s homotopy done={done} t={t} mu0={mu0}', flush=True)
            if done:
                print('final validation on normalized coordinates', flush=True)
                final = normalize(E)
                try:
                    d, mu, b, ok = validate_candidate(final, name, target)
                except CandidateRejected as exc:
                    print('rejected candidate:', exc, flush=True)
                    continue
                np.savetxt('%s/%s_equilateral_%dsticks.txt' % (out, name, target), final, fmt='%.17g')
                return ('FOUND equilateral %d-stick (defect %.1e, mu %.4f, bound %.1e, certified %s), %d reduction(s), %.0fs'
                        % (target, float(d), float(mu), float(b), ok, nreal, time.monotonic() - t0))
    return 'not found: %d %d-stick realization(s) in %.0fs' % (nreal, target, time.monotonic() - t0)

if __name__ == '__main__':
    names = sys.argv[1].split(',')
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    with open('../results/tenstick.log', 'a') as log:
        for nm in names:
            msg = '%-9s %s' % (nm, run(nm, budget))
            print(msg, flush=True)
            log.write(msg + '\n')
