"""
Created by Constantin Philippenko, 29th May 2024.
"""

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from skimage import data
from scipy.stats import ortho_group

from src.Client import ClientRealData, Client
from src.utilities.PytorchUtilities import get_mnist, get_celeba, get_real_sim, get_w8a


class Network:

    def __init__(self, nb_clients: int, nb_samples: int, dim: int, rank_S: int, plunging_dimension: int,
                 noise: int = 0, missing_value: int = 0, dataset_name: str = "synth", m=1, seed=1234):
        super().__init__()
        self.dataset_name = dataset_name
        self.m = m
        if dataset_name == "synth":
            self.nb_clients = nb_clients
            self.dim = dim
            self.plunging_dimension = plunging_dimension
            # self.mask = self.generate_mask(missing_value, nb_samples)
            self.rank_S = rank_S
            self.noise = noise
            self.clients = self.generate_network_of_clients(rank_S, missing_value, nb_samples, seed, noise)
            self.W = self.generate_neighborood()
            return
        elif dataset_name.__eq__("cameraman"):
            cameraman = data.camera()

            self.nb_clients = 1
            self.plunging_dimension = plunging_dimension

            self.dim = cameraman.shape[1]
            # self.mask = self.generate_mask(missing_value, nb_samples)

            self.clients = []
            for c_id in range(self.nb_clients):
                self.clients.append(ClientRealData(c_id, self.dim, cameraman.shape[0], cameraman, self.plunging_dimension,
                                    missing_value, noise))
            self.W = self.generate_neighborood()
        elif dataset_name in ["mnist", "celeba", "w8a", "real-sim"]:
            if dataset_name.__eq__("mnist"):
                dataset = get_mnist()
            elif dataset_name.__eq__("celeba"):
                dataset = get_celeba(nb_clients)
            elif dataset_name.__eq__("w8a"):
                dataset = get_w8a(nb_clients)
            elif dataset_name.__eq__("real-sim"):
                dataset = get_real_sim(nb_clients)
            else:
                raise ValueError(f"Not a correct dataset: {dataset_name}")

            self.nb_clients = len(dataset)
            self.plunging_dimension = plunging_dimension

            self.dim = dataset[0].shape[1]
            # self.mask = self.generate_mask(missing_value)

            self.clients = []
            for c_id in range(self.nb_clients):
                self.clients.append(ClientRealData(c_id, self.dim, dataset[c_id].shape[0], dataset[c_id],
                                                   self.plunging_dimension, missing_value, noise))
            self.W = self.generate_neighborood()

    def plot_graph_connectivity(self):
        if self.nb_clients == 1:
            return
        # We remove the self connection to avoid loop on the graph.
        G = nx.from_numpy_array(self.W - np.diag(np.diag(self.W)))
        # Draw the graph
        pos = nx.spring_layout(G)  # Positions nodes using Fruchterman-Reingold force-directed algorithm
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_weight='bold', font_size=10)
        plt.savefig(f"connectivity_graph_N{self.nb_clients}_r{self.plunging_dimension}.pdf", dpi=600,
                    bbox_inches='tight')

    def generate_neighborood(self, type: int = "REGULAR"):
        if self.nb_clients == 1:
            return np.array([[1]])
        if self.nb_clients == 2:
            return np.array([[0.5, 0.5], [0.5, 0.5]])
        if type == "REGULAR":
            W = nx.to_numpy_array(nx.random_regular_graph(4, self.nb_clients, seed=456)) + np.eye(self.nb_clients, dtype=int)
        elif type == "ERDOS":
            W = nx.to_numpy_array(nx.fast_gnp_random_graph(self.nb_clients, 0.5, seed=456)) + np.eye(self.nb_clients, dtype=int)
        else:
            raise ValueError("Unrecognized type of connectivity graph.")
        W = np.array(W / np.sum(W, axis=1))
        assert (np.sum(W, axis=1) == 1).all(), "W is not doubly stochastic."
        return W

    def generate_network_of_clients(self, rank_S: int, missing_value, nb_samples, seed, noise: int = 0):
        np.random.seed(151515)
        clients = []
        U_star, D_star, V_star = self.generate_low_rank_matrix(rank_S, nb_samples)
        S = U_star @ D_star @ V_star.T

        for c_id in range(self.nb_clients):
            S_i = S[c_id * nb_samples: (c_id + 1) * nb_samples]
            clients.append(Client(c_id, self.dim, nb_samples, S_i, self.plunging_dimension,
                                  missing_value, noise))
        return clients

    def generate_low_rank_matrix(self, rank: int, nb_samples):
        assert rank < self.dim, "The matrix rank must be smaller that the number of features d."

        V_star = ortho_group.rvs(dim=self.dim)[:rank].T
        U_star = ortho_group.rvs(dim=nb_samples * self.nb_clients)[:rank].T
        D_star = np.zeros((rank, rank))

        D_star[0, 0] = 1
        for k in range(1, rank):
            D_star[k, k] = 1

        return U_star, D_star, V_star