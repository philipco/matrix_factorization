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
        client.set_U(step_size * generate_gaussian_matrix(network.nb_samples, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))
        client.set_V(step_size * generate_gaussian_matrix(network.dim, plunging_dimension, 1) / (
            np.sqrt(plunging_dimension)))



def smart_MF_initialization(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0] # Tester avec sigma max au lieu de largest !
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size * largest_sv_S**2)))
        client.set_V(Phi_V * np.sqrt(step_size * largest_sv_S**2))
    return sigma_min, sigma_max

def smart_MF_initialization_for_GD_on_U(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_U = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_U(Phi_U * np.sqrt(step_size * largest_sv_S**2))
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size * largest_sv_S**2)))
    return sigma_min, sigma_max


def bi_smart_MF_initialization(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / np.sqrt(network.nb_samples))
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size * largest_sv_S**2)))
        client.set_V(client.S.T @ Phi_V * np.sqrt(step_size * largest_sv_S**2))
    return sigma_min, sigma_max

def bi_smart_MF_initialization_for_GD_on_U(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_V = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_U(client.S @ Phi_U * np.sqrt(step_size * largest_sv_S**2))
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size * largest_sv_S**2)))
    return sigma_min, sigma_max

def ortho_MF_initialization(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T * np.sqrt(network.dim)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_V = ortho_group.rvs(network.dim).T[:network.plunging_dimension].T
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size * largest_sv_S**2)))
        client.set_V(Phi_V * np.sqrt(step_size * largest_sv_S**2))
    return sigma_min, sigma_max

def ortho_MF_initialization_for_GD_on_U(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_V = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T * np.sqrt(network.nb_samples)
        key_matrix = network.clients[0].D_star @ network.clients[0].U_star.T @ Phi_V
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0]
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_U = ortho_group.rvs(network.nb_samples).T[:network.plunging_dimension].T #* np.sqrt(network.nb_samples)
        client.set_U(Phi_U * np.sqrt(step_size * largest_sv_S**2))
        client.set_V(client.S.T @ Phi_V / (np.sqrt(step_size * largest_sv_S**2)))
    return sigma_min, sigma_max


def smart_sparse_MF_initialization(network: Network, C: int, largest_sv_S: int):
    sigma_min = 0
    while sigma_min <= SINGULARVALUE_CLIP:
        Phi_U = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
        key_matrix = network.clients[0].D_star @ network.clients[0].V_star.T @ Phi_U
        sigma_min = svds(key_matrix, k=1, which='SM')[1][0]
        sigma_max = svds(key_matrix, k=1, which='LM')[1][0] # Tester avec sigma max au lieu de largest !
    print(f"===> kappa: {sigma_max/sigma_min}")
    print(f"===> sigma_min: {sigma_min}")
    step_size = 1 / sigma_max
    for client in network.clients:
        Phi_V = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / np.sqrt(network.dim))
        client.set_U(client.S @ Phi_U /(np.sqrt(step_size * largest_sv_S**2)))
        client.set_V(Phi_V * np.sqrt(step_size * largest_sv_S**2))
    return sigma_min, sigma_max
