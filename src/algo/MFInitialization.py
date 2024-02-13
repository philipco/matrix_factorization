import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import svds
from scipy.stats import ortho_group

from src.Client import Network
from src.MatrixUtilities import orth

SINGULARVALUE_CLIP = 0

def sparsify_matrix(M, p):
    for i in range(M.shape[0]):
        M[i] *= np.random.binomial(M.shape[1], p)
    return M/p


def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix

def random_power_iteration(network: Network):
    plunging_dimension = network.plunging_dimension
    for client in network.clients:
        client.set_initial_U(generate_gaussian_matrix(network.nb_samples, plunging_dimension, 1))
        client.set_initial_V(generate_gaussian_matrix(network.dim, plunging_dimension, 1))
    S = np.concatenate([client.U @ client.V.T for client in network.clients])
    largest_eigenvalues = svds(S, k=network.plunging_dimension, which='LM')[1]
    sigma_min = largest_eigenvalues[0]  # smallest non-zero eigenvalue
    sigma_max = largest_eigenvalues[-1]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def random_MF_initialization(network: Network):
    """Implementation of Global Convergence of Gradient Descent for Asymmetric Low-Rank Matrix Factorization, Ye and Du,
    Neurips 2021"""
    plunging_dimension = network.plunging_dimension
    S = np.concatenate([client.S for client in network.clients])
    largest_eigenvalues = svds(S, k=network.plunging_dimension, which='LM')[1]
    sigma_min = largest_eigenvalues[0]  # smallest non-zero eigenvalue
    sigma_max = largest_eigenvalues[-1]
    std = sigma_min / (np.sqrt(sigma_max * plunging_dimension **3 ) * (network.dim + network.nb_samples))
    for client in network.clients:
        client.set_initial_U(generate_gaussian_matrix(network.nb_samples, plunging_dimension, std))
        client.set_initial_V(generate_gaussian_matrix(network.dim, plunging_dimension, std))
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def smart_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    for client in network.clients:
        client.set_initial_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_initial_V(Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def smart_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        V += client.S.T @ Phi_V
        Phi_U = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_initial_U(Phi_U)

    #V = orth(V)
    key_matrix_for_condition_number = np.copy(V)
    for client in network.clients:
        client.set_initial_V(V)
    smallest_eigenvalues = svds(key_matrix_for_condition_number, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(key_matrix_for_condition_number, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def bi_smart_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    for client in network.clients:
        client.set_initial_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_initial_V(client.S.T @ Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def bi_smart_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        V += client.S.T @ Phi_V
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_initial_U(client.S @ Phi_U)

    #V = orth(V)
    key_matrix_for_condition_number = np.copy(V)
    for client in network.clients:
        client.set_initial_V(V)
    smallest_eigenvalues = svds(key_matrix_for_condition_number, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(key_matrix_for_condition_number, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def power_MF_initialization(network: Network, nb_power: int):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    for client in network.clients:
        U0 = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        for k in range(nb_power):
            U0 = client.S @ client.S.T @ U0 / client.dim
            U0 = orth(U0)
        client.set_initial_U(U0)
        U.append(U0)
        V = np.array([client.S.T @ U0[:, i] for i in range(network.plunging_dimension)]).T
        client.set_initial_V(V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def power_MF_initialization_for_GD_on_U(network: Network, nb_power: int):
    # assert nb_power >= 1, "There must be at least one power iteration."
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        V0 = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        V0 = client.S.T @ client.S @ V0
        V += V0

    #V = orth(V)
    key_matrix_for_condition_number = np.copy(V)
    for client in network.clients:
        U = client.S @ V
        client.set_initial_U(U)
        client.set_initial_V(V)
    largest_eigenvalues = svds(key_matrix_for_condition_number, k=network.plunging_dimension - 1, which='LM')[1]
    sigma_min = largest_eigenvalues[0]
    sigma_max = largest_eigenvalues[-1]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def ortho_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    Phi_U = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T * np.sqrt(network.dim)
    for client in network.clients:
        client.set_initial_U(client.S @ Phi_U)
        U.append(client.U)
        Phi_V = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T
        client.set_initial_V(Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def ortho_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    key_matrix_for_condition_number = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        Phi_V = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T * np.sqrt(network.nb_samples)
        V += client.S.T @ Phi_V
        key_matrix_for_condition_number += client.S_star.T @ Phi_V
        Phi_U = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T
        client.set_initial_U(Phi_U)

    #V = orth(V)
    key_matrix_for_condition_number = V
    for client in network.clients:
        client.set_initial_V(V)
    smallest_eigenvalues = svds(key_matrix_for_condition_number, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(key_matrix_for_condition_number, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def eig_MF_initialization(network: Network):
    U = []
    # When initialization U in the span of S, Phi is shared by all clients.
    # Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    for client in network.clients:
        w, v = eigh(client.S, subset_initial_by_index=[0, network.plunging_dimension-1])
        client.set_initial_U(v)
        U.append(client.U)
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_initial_V(Phi_V)
    U = np.concatenate(U)
    smallest_eigenvalues = svds(U, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
    sigma_max = svds(U, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def eig_MF_initialization_for_GD_on_U(network: Network):
    V = np.zeros((network.dim, network.plunging_dimension))
    for client in network.clients:
        w, v = eigh(client.S, subset_initial_by_index=[0, network.plunging_dimension-1])
        V += v
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_initial_U(Phi_U)
    for client in network.clients:
        client.set_initial_V(V)
    smallest_eigenvalues = svds(V, k=network.plunging_dimension - 1, which='SM')[1]
    sigma_min = np.min(smallest_eigenvalues[smallest_eigenvalues > 10**-10])  # smallest non-zero eigenvalue
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
        client.set_initial_U(client.S @ Phi_U)
        client.set_initial_V(Phi_V * 1)
    return sigma_min, sigma_max
