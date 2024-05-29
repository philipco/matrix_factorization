import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import svd
from scipy.linalg import eigh
from scipy.sparse.linalg import svds
from scipy.stats import ortho_group

from src.Client import Network
from src.MatrixUtilities import power, compute_svd

SINGULARVALUE_CLIP = 0


def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix

def random_power_iteration(network: Network):
    plunging_dimension = network.plunging_dimension
    for client in network.clients:
        client.set_initial_U(generate_gaussian_matrix(client.nb_samples, plunging_dimension, 1))
        client.set_initial_V(generate_gaussian_matrix(network.dim, plunging_dimension, 1))
    S = np.concatenate([client.S for client in network.clients])
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


def smart_MF_initialization_for_GD_on_U(network: Network):

    S = np.concatenate([client.S for client in network.clients])

    kappa_inf = np.inf
    for i in range(network.m):
        Phi_V = [generate_gaussian_matrix(client.nb_samples, client.plunging_dimension, 1) for client in
                 network.clients]
        V_clients = []
        # We compute (S.T @ S)^alpha @ S.T @ Phi in a distributed way to not compute and store a dxd matrix.
        for i in range(network.nb_clients):
            client = network.clients[i]
            V_clients.append(client.S.T @ Phi_V[i])

        V_sampled = np.sum([v for v in V_clients], axis=0)
        singular_values = compute_svd(V_sampled)
        sigma_max = singular_values[0]
        sigma_min = singular_values[network.rank_S - 1] if hasattr(network, "rank_S") else singular_values[-1]

        if kappa_inf > sigma_max / sigma_min:
            V = V_sampled
            kappa_inf = sigma_max / sigma_min
            Phi = np.concatenate(Phi_V)

    for i in range(network.nb_clients):
        network.clients[i].V = V

    for a in range(network.power):
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.V = client.S.T @ client.S @ V
        V = np.sum([client.V for client in network.clients], axis=0)

    V_centralized = power(S, alpha=network.power) @ Phi
    assert np.isclose(V, V_centralized).all(), "The distributed version of V is not correct."

    for client in network.clients:
        Phi_U = generate_gaussian_matrix(client.nb_samples, network.plunging_dimension,
                                         1 / np.sqrt(client.nb_samples))
        client.set_initial_U(Phi_U)
        client.set_initial_V(V)
    singular_values = compute_svd(V)
    sigma_max = singular_values[0]
    sigma_min = singular_values[network.rank_S-1] if hasattr(network, "rank_S") else singular_values[-1]
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max



