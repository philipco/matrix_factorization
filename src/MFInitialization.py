import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import ortho_group

from src.Client import Network

SINGULARVALUE_CLIP = 0

def sparsify_matrix(M, p):
    for i in range(M.shape[0]):
        M[i] *= np.random.binomial(M.shape[1], p)
    return M/p


def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix


def random_MF_initialization(network: Network):
    plunging_dimension = network.plunging_dimension
    step_size = 1
    for client in network.clients:
        client.set_U(generate_gaussian_matrix(network.nb_samples, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))
        client.set_V(generate_gaussian_matrix(network.dim, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))


def smart_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    for client in network.clients:
        client.set_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_V(Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def smart_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        V += client.S.T @ Phi_V
        Phi_U = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_U(Phi_U)
    for client in network.clients:
        client.set_V(V)
    smallest_eigenvalues = svds(V, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(V, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def bi_smart_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    for client in network.clients:
        client.set_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_V(client.S.T @ Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def bi_smart_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        V += client.S.T @ Phi_V
        Phi_U = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_U(client.S @ Phi_U)
    for client in network.clients:
        client.set_V(V)
    smallest_eigenvalues = svds(V, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(V, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def ortho_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T * np.sqrt(network.dim)
    for client in network.clients:
        client.set_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T
        client.set_V(Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    return sigma_min, sigma_max

def ortho_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T * np.sqrt(network.nb_samples)
        V += client.S.T @ Phi_V
        Phi_U = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T
        client.set_U(Phi_U)
    for client in network.clients:
        client.set_V(V)
    smallest_eigenvalues = svds(V, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[np.nonzero(smallest_eigenvalues)])  # smallest non-zero eigenvalue
    sigma_max = svds(V, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def smart_sparse_MF_initialization(network: Network):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0] # Tester avec sigma max au lieu de largest !
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_U(client.S @ Phi_U)
        client.set_V(Phi_V * 1)
    return sigma_min, sigma_max
