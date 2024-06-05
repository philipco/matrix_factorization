import numpy as np
from scipy.stats import ortho_group


def generate_low_rank_matrix(nb_clients: int, dim: int, rank: int, nb_samples):
    assert rank < dim, "The matrix rank must be smaller that the number of features d."

    V_star = ortho_group.rvs(dim=dim)[:rank].T
    U_star = ortho_group.rvs(dim=nb_samples * nb_clients)[:rank].T
    D_star = np.zeros((rank, rank))

    D_star[0, 0] = 1
    for k in range(1, rank):
        D_star[k, k] = 1

    return U_star, D_star, V_star

def sparsify_matrix(A, p):
    mask = np.random.choice([0, 1], size=A.shape, p=[1-p, p])
    return mask * A

def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix


def generate_sparse_random_matrix(n, d):
    sparse_random_matrix = np.random.choice(np.array([-np.sqrt(3), 0, np.sqrt(3)]), p=[1/6, 2/3, 1/6], size=(n, d))
    return sparse_random_matrix


def compute_svd(matrix, tol=10**-14):
    full_svd = np.linalg.svd(matrix, compute_uv=False)
    return full_svd[full_svd > tol]


def power(A, alpha):
    Apower = np.eye(A.shape[1], A.shape[1])
    for k in range(alpha):
        Apower = A.T @ A @ Apower
    return Apower @ A.T
