"""Exact Wirtinger propagation for rank-four Coxeter seed images.

The source gives each strand as a cyclic block from one undercrossing to the
next. All group arithmetic is on integers; no floating-point group checks.
"""
import ast
from functools import lru_cache
from itertools import product


def _strands(code):
    n = len(code)
    if n < 2 or n % 2 or any(type(x) is not int for x in code):
        raise ValueError('invalid Gauss sequence')
    crossings = n // 2
    if set(code) != set(range(1, crossings + 1)) | set(range(-crossings, 0)):
        raise ValueError('each crossing needs one over and one under visit')
    under = [i for i, value in enumerate(code) if value < 0]
    strands = []
    owner = {}
    starts = {}
    ends = {}
    for s, a in enumerate(under):
        b = under[(s + 1) % len(under)]
        width = (b - a) % n or n
        strand = tuple(code[(a + offset) % n] for offset in range(width + 1))
        strands.append(strand)
        starts[code[a]] = s
        ends[code[b]] = s
        for offset in range(width):
            value = code[(a + offset) % n]
            if value > 0:
                owner[value] = s
    return strands, starts, ends, owner


def _transposition(value):
    pair = ast.literal_eval(value)
    if (not isinstance(pair, tuple) or len(pair) != 2 or
            any(type(x) is not int or x < 1 or x > 5 for x in pair) or pair[0] == pair[1]):
        raise ValueError(f'not an S5 transposition: {value}')
    return tuple(sorted(pair))


def _s5_conjugate(over, under):
    x, y = over
    return tuple(sorted((y if z == x else x if z == y else z for z in under)))


def _s5_order(generators):
    identity = tuple(range(1, 6))
    swaps = []
    for a, b in generators:
        p = list(identity)
        p[a - 1], p[b - 1] = b, a
        swaps.append(tuple(p))
    seen = {identity}
    frontier = [identity]
    for p in frontier:
        for q in swaps:
            product = tuple(p[q[i] - 1] for i in range(5))
            if product not in seen:
                seen.add(product)
                frontier.append(product)
    return len(seen)


_D4_GRAM = ((2, -1, 0, 0), (-1, 2, -1, -1),
            (0, -1, 2, 0), (0, -1, 0, 2))
_D4_IDENTITY = tuple(int(i == j) for i in range(4) for j in range(4))


def _d4_product(a, b):
    return tuple(sum(a[4*i+k] * b[4*k+j] for k in range(4))
                 for i in range(4) for j in range(4))


def _d4_closure(generators):
    seen = {_D4_IDENTITY}
    frontier = [_D4_IDENTITY]
    for a in frontier:
        for b in generators:
            product = _d4_product(a, b)
            if product not in seen:
                seen.add(product)
                frontier.append(product)
    return frozenset(seen)


@lru_cache(maxsize=1)
def _d4_weyl_group():
    """W(D4) from simple-root reflections in the D4 Cartan basis."""
    simple = []
    for i in range(4):
        simple.append(tuple((int(j == k) - _D4_GRAM[i][k]) if j == i
                            else int(j == k) for j in range(4) for k in range(4)))
    group = _d4_closure(simple)
    if len(group) != 192:
        raise AssertionError('incorrect canonical D4 group construction')
    return group


@lru_cache(maxsize=1)
def _d4_root_reflections():
    """The 12 matrices I - alpha (alpha^T Q) for D4 roots of norm 2."""
    reflections = set()
    for alpha in product(range(-2, 3), repeat=4):
        covector = tuple(sum(alpha[k] * _D4_GRAM[k][j] for k in range(4))
                         for j in range(4))
        if sum(alpha[j] * covector[j] for j in range(4)) != 2:
            continue
        reflections.add(tuple(int(i == j) - alpha[i] * covector[j]
                              for i in range(4) for j in range(4)))
    if len(reflections) != 12 or not reflections <= _d4_weyl_group():
        raise AssertionError('incorrect D4 root reflections')
    return frozenset(reflections)


def _d4_matrix(value):
    rows = tuple(ast.literal_eval(part) for part in value.split('|'))
    if (len(rows) != 4 or any(not isinstance(row, tuple) or len(row) != 4 for row in rows)
            or any(type(x) is not int for row in rows for x in row)):
        raise ValueError('D4 image is not a four-by-four integer matrix')
    matrix = tuple(x for row in rows for x in row)
    if matrix not in _d4_root_reflections() or _d4_product(matrix, matrix) != _D4_IDENTITY:
        raise ValueError('D4 seed is not a root reflection in the canonical Weyl group')
    return matrix


def _d4_conjugate(over, under):
    return _d4_product(_d4_product(over, under), over)


def verify_map(gauss_text, seed_text, kind):
    """Return a reproducible, exact relation/generation check for the seed map."""
    if kind not in ('S5', 'D4'):
        raise ValueError(f'unsupported Coxeter group: {kind}')
    parse = _transposition if kind == 'S5' else _d4_matrix
    conjugate = _s5_conjugate if kind == 'S5' else _d4_conjugate
    code = ast.literal_eval(gauss_text)
    strands, starts, ends, owner = _strands(code)
    raw = ast.literal_eval(seed_text)
    if not isinstance(raw, dict) or len(raw) != 4:
        raise ValueError('expected exactly four seed images')
    lookup = {str(strand): index for index, strand in enumerate(strands)}
    images = {}
    for key, value in raw.items():
        if key not in lookup or lookup[key] in images:
            raise ValueError(f'unknown or repeated seed strand: {key}')
        images[lookup[key]] = parse(value)
    expected = len(strands)
    while len(images) < expected:
        changed = False
        for k in range(1, expected + 1):
            left, right, over = ends[-k], starts[-k], owner[k]
            if over not in images:
                continue
            if left in images and right not in images:
                images[right] = conjugate(images[over], images[left])
                changed = True
            elif right in images and left not in images:
                images[left] = conjugate(images[over], images[right])
                changed = True
        if not changed:
            break
    mismatches = []
    checked = 0
    for k in range(1, expected + 1):
        left, right, over = ends[-k], starts[-k], owner[k]
        if all(s in images for s in (left, right, over)):
            checked += 1
            if images[right] != conjugate(images[over], images[left]):
                mismatches.append(k)
    generators = [parse(v) for v in raw.values()]
    order = (_s5_order(generators) if kind == 'S5' else len(_d4_closure(generators)))
    generates = (order == 120 if kind == 'S5' else
                 _d4_closure(generators) == _d4_weyl_group())
    passed = len(images) == expected and checked == expected and not mismatches and generates
    status = ('pass' if passed else 'incomplete_labeling' if checked != expected else
              'relation_failure' if mismatches else 'not_surjective')
    return {'passed': passed, 'status': status, 'generation_passed': generates,
            'group': kind, 'strand_count': expected,
            'strands_labeled': len(images), 'relations_checked': checked,
            'failed_relations': mismatches, 'group_order': order,
            'strand_images': {str(strands[i]): str(images[i]) for i in sorted(images)}}
