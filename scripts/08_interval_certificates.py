"""Run geometric interval certificates on stored decimal polygon coordinates.

From the repository root:
  python scripts/08_interval_certificates.py --expected-count 25

The output does NOT certify knot identity or the stick number of any knot.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equistick.interval_certificate import certify_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'interval_certificates.json')
    parser.add_argument('--precision-bits', type=int, default=160)
    parser.add_argument('--max-pairs', type=int, default=1000)
    parser.add_argument('--timeout-s', type=float, default=30)
    parser.add_argument('--expected-count', type=int)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s < 0:
        parser.error('timeout must be finite and nonnegative')
    paths = sorted(args.results_dir.glob('*_equilateral_*sticks.txt'))
    if not paths:
        parser.error('no coordinate files found')
    if args.expected_count is not None and len(paths) != args.expected_count:
        parser.error(f'expected {args.expected_count} coordinate files, found {len(paths)}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source = ROOT / 'equistick' / 'interval_certificate.py'
    dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'equistick/interval_certificate.py',
         'scripts/08_interval_certificates.py'], cwd=ROOT, text=True).strip())
    files = [certify_file(path, precision_bits=args.precision_bits,
                          max_pairs=args.max_pairs, timeout_s=args.timeout_s) for path in paths]
    report = {
        'certificate_scope': 'geometric MR inequality only; not knot identity',
        'method': 'exact decimal rational quadratic segment minima; outward mpmath.iv bounds',
        'code_revision': revision,
        'code_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'batch_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checker_dirty': dirty,
        'precision_bits': args.precision_bits,
        'max_pairs_per_file': args.max_pairs,
        'timeout_seconds_per_file': args.timeout_s,
        'file_count': len(files),
        'certified_count': sum(row['status'] == 'certified' for row in files),
        'files': files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                     prefix='.interval_certificate_', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    try:
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{report['certified_count']}/{report['file_count']} geometric MR inequalities certified; {args.output}")
    return 0 if report['certified_count'] == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())
