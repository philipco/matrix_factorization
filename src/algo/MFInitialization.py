"""
Created by Constantin Philippenko, 11th December 2023.
"""

import numpy as np
from scipy.sparse.linalg import svds

from src.Network import Network
from src.utilities.MatrixUtilities import power, compute_svd, generate_gaussian_matrix

SINGULARVALUE_CLIP = 0


def ward_and_kolda_init(network: Network):
    S = np.concatenate([client.S for client in network.clients])
    largest_eigenvalues = svds(S, k=network.plunging_dimension, which='LM')[1]
    sigma_min = largest_eigenvalues[0]  # smallest non-zero eigenvalue
    sigma_max = largest_eigenvalues[-1]
    total_nb_points = np.sum([client.nb_samples for client in network.clients])

    kappa = np.inf
    for i in range(20):
        PhiV = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / network.plunging_dimension)
        Phi = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / network.dim)
        product = S @ Phi
        eigenvalues = compute_svd(product)
        if eigenvalues[0] / eigenvalues[-1] < kappa:
            PhiU = Phi

    eta = 1 / sigma_max**2
    for client in network.clients:
        client.U = client.S @ PhiU / (sigma_max**2 * eta**0.5)
        client.V = eta**0.5 * sigma_max * PhiV
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max

def random_power_iteration(network: Network):
    """Random initialisation of U and V as two Gaussian matrices."""
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
    """Implementation of the random initialisation of U,V.
    From Global Convergence of Gradient Descent for Asymmetric Low-Rank Matrix Factorization, Ye and Du,
    Neurips 2021"""
    plunging_dimension = network.plunging_dimension
    S = np.concatenate([client.S for client in network.clients])
    largest_eigenvalues = svds(S, k=network.plunging_dimension, which='LM')[1]
    sigma_min = largest_eigenvalues[0]  # smallest non-zero eigenvalue
    sigma_max = largest_eigenvalues[-1]
    std = sigma_min / (np.sqrt(sigma_max * plunging_dimension ** 3) * (
            network.dim + np.sum([client.nb_samples for client in network.clients])))
    for client in network.clients:
        # std = sigma_min / (np.sqrt(sigma_max * plunging_dimension ** 3) * (
        #             network.dim + client.nb_samples))
        client.set_initial_U(generate_gaussian_matrix(client.nb_samples, plunging_dimension, std))
        client.set_initial_V(generate_gaussian_matrix(client.dim, plunging_dimension, std))
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def random_MF_initialization(network: Network):
    """Implementation of the random initialisation of U,V.
    From Global Convergence of Gradient Descent for Asymmetric Low-Rank Matrix Factorization, Ye and Du,
    Neurips 2021"""
    plunging_dimension = network.plunging_dimension
    S = np.concatenate([client.S for client in network.clients])
    largest_eigenvalues = svds(S, k=network.plunging_dimension, which='LM')[1]
    sigma_min = largest_eigenvalues[0]  # smallest non-zero eigenvalue
    sigma_max = largest_eigenvalues[-1]
    std = sigma_min**2 / (np.sqrt(sigma_max**2 * plunging_dimension ** 3) * (
            network.dim + np.sum([client.nb_samples for client in network.clients])))
    for client in network.clients:
        # std = sigma_min / (np.sqrt(sigma_max * plunging_dimension ** 3) * (
        #             network.dim + client.nb_samples))
        client.set_initial_U(generate_gaussian_matrix(client.nb_samples, plunging_dimension, std))
        client.set_initial_V(generate_gaussian_matrix(client.dim, plunging_dimension, std))
    print(f"===> kappa: {sigma_max / sigma_min}")
    return sigma_min, sigma_max


def distributed_power_initialization_for_GD_on_U(network: Network):
    """Implement the distributed power initialisation as presented in our paper."""

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



