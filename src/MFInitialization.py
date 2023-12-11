import numpy as np

from src.Client import Network


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


def smart_MF_initialization(network: Network, step_size, C: int, D: int):
    Phi = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1)
    Phi_prime = generate_gaussian_matrix(network.dim, network.plunging_dimension, 1 / network.dim)
    for client in network.clients:
        client.set_U(client.S @ Phi /(np.sqrt(step_size) * C))
        client.set_V(Phi_prime * np.sqrt(step_size) * D)

def smart_MF_initialization_for_GD_on_U(network: Network, step_size, C: int, D: int):
    Phi = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1 / network.nb_samples)
    Phi_prime = generate_gaussian_matrix(network.nb_samples, network.plunging_dimension, 1)
    for client in network.clients:
        client.set_U(Phi * np.sqrt(step_size) * D)
        client.set_V(client.S.T @ Phi_prime / (np.sqrt(step_size) * C))

