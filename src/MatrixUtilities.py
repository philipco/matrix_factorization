import numpy as np

def orth(A):
    """
    Gram-Schmidt orthogonalization of the columns of matrix A.

    Parameters:
    - A: Input matrix with linearly independent columns.

    Returns:
    - Q: Orthogonal matrix whose columns form an orthonormal basis.
    """
    m, n = A.shape
    Q = np.zeros((m, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            q = Q[:, i]
            v -= np.dot(v, q) * q
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:  # Avoid division by zero
            Q[:, j] = v / norm_v
        else:
            Q[:, j] = v

    return Q
