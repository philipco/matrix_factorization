import numpy as np


def compute_svd(matrix, tol=10**-14):
    full_svd = np.linalg.svd(matrix, compute_uv=False)
    return full_svd[full_svd > tol]


def power(A, alpha):
    Apower = np.eye(A.shape[1], A.shape[1])
    for k in range(alpha):
        Apower = A.T @ A @ Apower
    return Apower @ A.T
