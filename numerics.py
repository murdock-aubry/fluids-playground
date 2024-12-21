import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def create_rotated_ellipse_mask(width, height, angle_degrees):
    x = np.linspace(-1, 1, width)  # Horizontal axis, normalized
    y = np.linspace(-1, 1, height)  # Vertical axis, normalized
    X, Y = np.meshgrid(x, y)  # Create meshgrid for plotting
    
    # Define the ellipse equation parameters
    a = 0.3  # Semi-major axis (horizontal)
    b = 0.05  # Semi-minor axis (vertical)

    # Convert angle to radians for rotation
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    
    # Flatten the grid coordinates and apply the rotation
    coords = np.vstack([X.ravel(), Y.ravel()])
    rotated_coords = rotation_matrix @ coords

    # Reshape back to the grid
    X_rot = rotated_coords[0, :].reshape(X.shape)
    Y_rot = rotated_coords[1, :].reshape(Y.shape)
    
    # Define the ellipse equation: (X_rot^2 / a^2) + (Y_rot^2 / b^2) <= 1
    mask = (X_rot**2 / a**2 + Y_rot**2 / b**2) <= 1
    
    return mask.astype(int)


def construct_sparse_matrix(div, shape):
    h, w = shape
    diagonals = []
    offsets = []

    # Central diagonal
    center = 4 * np.ones(div.size)
    diagonals.append(center)
    offsets.append(0)

    # Horizontal neighbors
    diagonals.append(-np.ones(div.size - 1))
    offsets.append(1)
    diagonals.append(-np.ones(div.size - 1))
    offsets.append(-1)

    # Vertical neighbors
    diagonals.append(-np.ones(div.size - (w - 2)))
    offsets.append(w - 2)
    diagonals.append(-np.ones(div.size - (w - 2)))
    offsets.append(-(w - 2))

    # Construct sparse matrix A
    A = diags(diagonals, offsets, shape=(div.size, div.size))
    A = A.tocsr()  # Make it usable in the solver

    return A
