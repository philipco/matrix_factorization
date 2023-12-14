import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import ortho_group

from src.Client import Network

SINGULARVALUE_CLIP = 3

def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix


def random_MF_initialization(network: Network, step_size):
    plunging_dimension = network.plunging_dimension
    for client in network.clients:
        client.set_U(step_size * generate_gaussian_matrix(network.nb_samples, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))
        client.set_V(step_size * generate_gaussian_matrix(network.dim, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))


def smart_MF_initialization(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= 2:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / network.dim)
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size) * C))
        client.set_V(Phi_V * np.sqrt(step_size) * D)

def smart_MF_initialization_for_GD_on_U(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= 2:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_U = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / network.nb_samples)
        client.set_U(Phi_U * np.sqrt(step_size) * D)
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size) * C))


def bi_smart_MF_initialization(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= 2:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / network.nb_samples)
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size) * C))
        client.set_V(client.S.T @ Phi_V * np.sqrt(step_size) * D)

def bi_smart_MF_initialization_for_GD_on_U(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= 2:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / network.dim)
        client.set_U(client.S @ Phi_U * np.sqrt(step_size) * D)
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size) * C))

def ortho_MF_initialization(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T * np.sqrt(network.dim)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_V = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size) * C))
        client.set_V(Phi_V * np.sqrt(step_size) * D)

def ortho_MF_initialization_for_GD_on_U(network: Network, step_size, C: int, D: int, largest_sv_S: int):
    C *= largest_sv_S
    D *= largest_sv_S
    sigma_min = 0
    while sigma_min <= 2:
        Phi_V = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T * np.sqrt(network.nb_samples)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    for client in network.clients:
        Phi_U = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T #* np.sqrt(network.nb_samples)
        client.set_U(Phi_U * np.sqrt(step_size) * D)
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size) * C))


