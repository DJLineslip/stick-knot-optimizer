import numpy as np
import sys

filename = "run_4_simplified_9_29_knot_11_20241227_102627_.xyz"

def load_data_with_indices(filename):
    """
    Load 3D coordinates from a .xyz file and assign an index to each point.

    Each line in the file should contain three floating-point numbers
    separated by spaces, representing the x, y, and z coordinates.

    Parameters:
        filename (str): Path to the .xyz file.

    Returns:
        list: A list of lists, where each sublist contains [index, x, y, z].
    """
    try:
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Handle single line files
        if data.shape[1] != 3:
            raise ValueError(f"Expected 3 columns in the .xyz file, got {data.shape[1]}")
        original_data = [[i + 1, *point] for i, point in enumerate(data)]
        print(original_data)
        return original_data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

# Call the function to execute it
load_data_with_indices(filename)