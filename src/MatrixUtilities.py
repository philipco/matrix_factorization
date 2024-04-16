import numpy as np


def power(A, alpha):
    Apower = np.eye(A.shape[1], A.shape[1])
    for k in range(alpha):
        Apower = A.T @ A @ Apower
    return Apower @ A.T
