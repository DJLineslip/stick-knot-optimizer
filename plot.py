#!/usr/bin/env python3
"""
Plot results in 3D
"""

import numpy as np
import random
import sys
import datetime

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

simplified_data = [
    [1,  11.661284,-0.287630,  6.732491],
    [3,  4.628821,  13.065284, 7.436309],
    [4,  0.050201,  2.228618,  2.854664],
    [5,  7.515718,  1.702029,  6.475162],
    [6,  2.543000,  9.006243,  2.605009],
    [7,  2.568328,  8.450995,  7.074828],
    [8,  5.531225,  10.483731, 1.224715],
    [9,  5.952713,  4.639598,  5.598433],
    [10, 5.489934, -1.125123,  5.079889],
    [11, 7.243665,  9.856205,  3.187521],
    [12, 0.148528,  1.761894,  3.771345]
]

simplified_data = [
[1, 2.767318, 14.775825, 6.267689],
[1, 1.347820, 3.086494, 3.343214],
[1, 4.493542, 0.691664, 8.423534],
[1, 6.683900, 0.938069, 7.347526],
[1, 1.107233, 7.811968, 3.132928],
[1, 2.550999, 9.216473, 6.707692],
[1, 5.068590, 10.340801, 2.568143],
[1, 6.983436, 4.030579, 7.741478],
[1, 4.565627, -0.679322, 6.727437],
[1, -5.780230, 6.348016, 2.737210],
[1, 6.430940, 11.545038, 3.725138],
[1, -0.745798, -0.062917, 5.009586]
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
        raise RuntimeError("pyknotid not installed. Cannot identify knot type.")

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
# 9) Main
#####################

def main():
    if not HAS_PYKNOTID:
        print("WARNING: pyknotid not installed. The script will fail on 'is_still_929_knot()' calls.")
        sys.exit(1)  # Exit as the rest cannot proceed without pyknotid

    # 1) Load polygon
    polygon = get_initial_polygon(simplified_data)
    is_still_929_knot(polygon)
    print("Initial polygon shape:", polygon.shape)

    plot_polygon_3d(polygon, title="Polygon")

if __name__ == "__main__":
    main()
