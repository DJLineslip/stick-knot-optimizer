#!/usr/bin/env python3
"""
Production-style pipeline to:

1. Load a 9_29 knot with 13 edges
2. Attempt to reduce it to 9 edges via local moves (while checking knot type w/ pyknotid)
3. Agitate if stuck
4. Attempt equilateral optimization (gradient-based)
5. Plot results in 3D
"""

import numpy as np
import random
import sys

# If you have pyknotid installed:
try:
    from pyknotid.spacecurves import Knot
    HAS_PYKNOTID = True
except ImportError:
    HAS_PYKNOTID = False

#############################
# 1) Hardcode / Load the Data
#############################

original_data = [
    [1,   8.833, -4.108,  0.448],
    [2,  -1.662,  8.632,  0.986],
    [3,  -4.043, -1.997, -1.967],
    [4,  -0.453, -4.890,  2.602],
    [5,   3.209, -4.030,  0.726],
    [6,  -4.006,  4.028, -1.311],
    [7,  -2.130,  3.765,  2.452],
    [8,   0.433,  4.356, -2.663],
    [9,   1.947, -1.129,  1.325],
    [10, -0.0376,-6.077,  1.294],
    [11, -10.742, 2.272, -1.653],
    [12,  1.873,  6.465, -1.644],
    [13, -5.543, -4.438, -0.800]
]

def get_initial_polygon(data):
    """
    Returns a NumPy array of shape (n, 3) of the points,
    ignoring the first 'index' column, in ascending order by index.
    """
    data_sorted = sorted(data, key=lambda row: row[0])
    coords = [row[1:] for row in data_sorted]
    return np.array(coords, dtype=float)

#########################
# 2) Knot Identification
#########################

def is_still_929_knot(vertices):
    """
    Attempt to identify the knot type using pyknotid.
    Return True if it identifies as 9_29, otherwise False.

    If pyknotid is not installed, we raise an exception.
    """
    if not HAS_PYKNOTID:
        raise RuntimeError("pyknotid not installed. Cannot identify knot type in production code.")

    knot = Knot(vertices, verbose=False)

    # Call the identify() method
    identified_types = knot.identify()

    # Debugging: Print identified types
    print(f"Identified types: {identified_types}")  # For debugging

    # Check if '9_29' or '9n29' is among the identified types
    if isinstance(identified_types, list):
        for kt in identified_types:
            # Access the 'name' attribute of the Knot object
            kt_str = str(kt)
            if '9_29' in kt_str or '9n29' in kt_str:
                return True
    elif isinstance(identified_types, str):
        if '9_29' in identified_types or '9n29' in identified_types:
            return True

    return False

###############################
# 3) Local Moves & Simplification
###############################

def remove_vertex_if_possible(vertices, idx):
    """
    Remove vertex #idx by connecting its neighbors, with a small jiggle.
    Then check for intersection and knot type.

    Returns a new (n-1) array if successful, else None.
    """
    n = len(vertices)
    prev_idx = (idx - 1) % n
    next_idx = (idx + 1) % n

    # small random jiggle
    jiggle_scale = 0.01
    random_perturbation = jiggle_scale * np.random.rand(3)
    vertices[prev_idx] += random_perturbation

    # Remove 'idx'
    new_vertices = np.delete(vertices, idx, axis=0)

    # Map old-index -> new-index for neighbor
    old_next = next_idx
    if old_next > idx:
        new_next = old_next - 1
    else:
        new_next = old_next

    random_pert2 = jiggle_scale * np.random.rand(3)
    new_vertices[new_next] += random_pert2

    # Check intersection:
    if has_forbidden_intersection(new_vertices):
        print(f"Vertex {idx} removal caused forbidden intersection.")
        return None

    # Check knot:
    if not is_still_929_knot(new_vertices):
        print(f"Vertex {idx} removal changed knot type.")
        return None

    print(f"Vertex {idx} successfully removed. New vertex count: {len(new_vertices)}")
    return new_vertices

def simplify_polygon(vertices, target_edges=9, max_iterations=1000):
    """
    Repeatedly attempt to remove vertices until we reach 'target_edges'
    or can't remove more without changing the knot type.
    """
    current = vertices.copy()
    n = len(current)
    iteration = 0
    while n > target_edges and iteration < max_iterations:
        success = False
        for idx in range(n):
            candidate = remove_vertex_if_possible(current, idx)
            if candidate is not None:
                current = candidate
                n = len(current)
                success = True
                break
        if not success:
            # No vertex was removable without changing type or causing intersection
            # -> we do a random agitation and try again
            current = agitate_polygon(current)
        iteration += 1
    return current

##################
# 4) Agitation
##################

def agitate_polygon(vertices, scale=0.1, attempts=20):
    """
    Randomly move each vertex slightly to free geometry,
    hopefully enabling further local moves.
    """
    best_polygon = vertices.copy()
    for _ in range(attempts):
        candidate = vertices.copy()
        n = len(candidate)
        # random perturbation on all vertices
        for i in range(n):
            perturb = np.random.uniform(-scale, scale, size=3)
            candidate[i] += perturb
        if not has_forbidden_intersection(candidate):
            # Check if it remains 9_29
            if is_still_929_knot(candidate):
                best_polygon = candidate
                print("Agitation successful: Knot type preserved after perturbation.")
                break
    return best_polygon

############################
# 5) Equilateral Optimization
############################

def enforce_equilateral(vertices, max_steps=1000, alpha=1.0, beta=1.0, step_size=0.01):
    """
    Attempt to make edges the same length by gradient descent on:
        E_eq = sum (L_i - L_avg)^2
    plus a self-avoidance term E_avoid, done node-to-node for simplicity.

    alpha, beta are weighting factors for eq vs. avoid.

    If the polygon unknots, returns None.
    """
    current = vertices.copy()
    for step in range(max_steps):
        grad_eq = gradient_eq_length(current)
        grad_avoid = gradient_self_avoidance(current)

        total_grad = alpha * grad_eq + beta * grad_avoid

        current = current - step_size * total_grad

        # Check if we still have 9_29
        if not is_still_929_knot(current):
            print("Equilateral optimization changed knot type.")
            return None  # Possibly unknotted or changed type

        # Check small gradient => convergence
        if np.linalg.norm(total_grad) < 1e-5:
            print(f"Equilateral optimization converged after {step} steps.")
            break
    return current

def gradient_eq_length(vertices):
    """
    Compute gradient of E_eq = sum_i (L_i - L_avg)^2,
    ignoring the derivative of L_avg w.r.t. each L_i for simplicity.

    That is:
      L_i = ||v_{i+1} - v_i||,
      delta_i = (L_i - L_avg),
      partial(L_i) / partial(v_i) = (v_i - v_{i+1}) / L_i,
      partial(L_i) / partial(v_{i+1}}) = (v_{i+1} - v_i) / L_i.

    We'll do:
      grad E_eq wrt v_i = sum_{edges that involve i} 2 * delta_i * partial(L_i)/partial(v_i).
    """
    n = len(vertices)
    # 1) Compute all edge lengths
    lengths = []
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i+1)%n]
        lengths.append(np.linalg.norm(p2 - p1))

    L_avg = np.mean(lengths)
    grad = np.zeros_like(vertices)

    # 2) For each edge i, add contribution to v_i and v_{i+1}
    for i in range(n):
        L_i = lengths[i]
        if L_i < 1e-12:
            # degenerate edge, skip or handle separately
            continue
        delta_i = (L_i - L_avg)
        # partial wrt v_i
        direction_i = (vertices[i] - vertices[(i+1)%n]) / L_i  # derivative of L_i wrt v_i
        # partial wrt v_{i+1}
        direction_ip1 = (vertices[(i+1)%n] - vertices[i]) / L_i

        grad[i]     += 2.0 * delta_i * direction_i
        grad[(i+1)%n] += 2.0 * delta_i * direction_ip1

    return grad

def gradient_self_avoidance(vertices, power=2):
    """
    Compute a simple node-based 'self-avoidance' gradient:
    E_avoid = sum_{i<j} 1 / ||v_i - v_j||^2

    For each pair (i,j), partial wrt v_i:
        d/dv_i of 1/dist^2 = -2 (v_i - v_j) / dist^4
    We'll accumulate these in an O(n^2) loop.
    """
    n = len(vertices)
    grad = np.zeros_like(vertices)
    for i in range(n):
        for j in range(i+1, n):
            diff = vertices[i] - vertices[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                # extremely close or identical points; skip or handle
                continue
            # E_ij = 1 / dist_sq
            # partial wrt v_i: -2 (v_i - v_j) / dist_sq^2
            inv_dist4 = 1.0 / (dist_sq * dist_sq)
            common = -2.0 * inv_dist4
            grad[i] += common * diff     # (v_i - v_j)
            grad[j] -= common * diff     # by symmetry
    return grad

######################
# 6) Intersection Check
######################

def has_forbidden_intersection(vertices):
    """
    Check if there's a 3D self-intersection among edges,
    i.e. any pair of edges is closer than a small tol, or truly crossing.
    """
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i+1)%n]
        for j in range(i+2, n):
            # skip adjacent edges or the same edge
            if j in [(i-1)%n, i, (i+1)%n]:
                continue
            p3 = vertices[j]
            p4 = vertices[(j+1)%n]
            if segments_intersect_3d(p1, p2, p3, p4, tol=1e-3):
                return True
    return False

def segments_intersect_3d(a1, a2, b1, b2, tol=1e-5):
    dist = segment_segment_distance_3d(a1, a2, b1, b2)
    return (dist < tol)

def segment_segment_distance_3d(p1, p2, p3, p4):
    """
    Returns the minimal distance between two 3D line segments p1->p2 and p3->p4.

    Implementation adapted from http://geomalgorithms.com/a07-_distance.html
    """
    EPS = 1e-12

    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = np.dot(u,u)  # squared length of u
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    D = a*c - b*b
    sc, sN, sD = D, D, D  # sc = sN / sD
    tc, tN, tD = D, D, D  # tc = tN / tD

    # Compute the line parameters of the two closest points
    if D < EPS:
        # the lines are almost parallel
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        # get the closest points on the infinite lines
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        # tc > 1 => the t=1 edge is visible
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    # finally do the division to get sc and tc
    sc = 0.0 if abs(sN) < EPS else sN / sD
    tc = 0.0 if abs(tN) < EPS else tN / tD

    # get the difference of the two closest points
    # = w + sc*u - tc*v
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

#####################
# 7) Plotting
#####################

def plot_polygon_3d(vertices, title=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    # close the loop
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    z_closed = np.append(z, z[0])

    ax.plot(x_closed, y_closed, z_closed, 'o-', label='Knot')
    ax.set_box_aspect((1,1,1))  # requires mpl>=3.2 for equal aspect

    if title:
        ax.set_title(title)
    ax.legend()
    plt.show()

#####################
# 8) Main
#####################

def main():
    if not HAS_PYKNOTID:
        print("WARNING: pyknotid not installed. The script will fail on 'is_still_929_knot()' calls.")
        sys.exit(1)  # Exit as the rest cannot proceed without pyknotid

    # 1) Load initial polygon (13 edges)
    initial_polygon = get_initial_polygon(original_data)
    print("Initial polygon shape:", initial_polygon.shape)

    # 2) Simplify to 9 edges
    simplified = simplify_polygon(initial_polygon, target_edges=9, max_iterations=200)
    if len(simplified) == 9:
        print("Successfully simplified to 9 edges!")
    else:
        print("WARNING: Did not reach 9 edges. Current #edges =", len(simplified))

    plot_polygon_3d(simplified, title="Simplified Polygon")

    # 3) Attempt equilateral optimization
    eq_result = enforce_equilateral(simplified, max_steps=5000, alpha=1.0, beta=0.5, step_size=0.01)
    if eq_result is not None:
        print("Equilateral optimization converged (or terminated).")
        # Evaluate final edge lengths
        final_lengths = []
        n = len(eq_result)
        for i in range(n):
            p1 = eq_result[i]
            p2 = eq_result[(i+1)%n]
            final_lengths.append(np.linalg.norm(p2 - p1))
        stdev = np.std(final_lengths)
        print("Edge lengths after optimization:", final_lengths)
        print(f"Std dev of edge lengths: {stdev:.6f}")
        plot_polygon_3d(eq_result, title="Equilateral Attempt")
    else:
        print("Equilateral optimization changed knot type or got stuck (None returned).")

if __name__ == "__main__":
    main()
