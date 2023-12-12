"""
Created by Constantin Philippenko, 11th December 2023.
"""
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds
from scipy.stats import ortho_group


class Network:

    def __init__(self, nb_clients: int, nb_samples: int, dim: int, rank_S: int, plunging_dimension: int):
        super().__init__()
        self.nb_clients = nb_clients
        self.dim = dim
        self.plunging_dimension = plunging_dimension
        self.nb_samples = nb_samples
        self.clients = self.generate_network_of_clients(rank_S)
        self.W = self.generate_neighborood()
        self.plot_graph_connectivity()

    def plot_graph_connectivity(self):
        # We remove the self connection to avoid loop on the graph.
        G = nx.from_numpy_matrix(self.W - np.eye(self.nb_clients, dtype=int))
        # Draw the graph
        pos = nx.spring_layout(G)  # Positions nodes using Fruchterman-Reingold force-directed algorithm
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_weight='bold', font_size=10)
        plt.savefig(f"connectivity_graph_N{self.nb_clients}_r{self.plunging_dimension}.pdf", dpi=600,
                    bbox_inches='tight')

    def generate_neighborood(self, type: int = "REGULAR"):
        if type == "REGULAR":
            return nx.to_numpy_matrix(nx.random_regular_graph(4, self.nb_clients, seed=456)) + np.eye(self.nb_clients, dtype=int)
        elif type == "ERDOS":
            return nx.to_numpy_matrix(nx.fast_gnp_random_graph(self.nb_clients, 0.5, seed=456)) + np.eye(self.nb_clients, dtype=int)
        else:
            raise ValueError("Unrecognized type of connectivity graph.")

    def generate_network_of_clients(self, rank_S: int):
        clients = []
        V_star = ortho_group.rvs(dim=self.dim)
        for c_id in range(self.nb_clients):
            S_star = self.generate_low_rank_matrix(V_star, rank_S)
            clients.append(Client(c_id, self.dim, self.nb_samples, S_star, self.plunging_dimension))
        return clients

    def generate_low_rank_matrix(self, V_star, rank: int):
        assert self.dim <= self.nb_samples, "The numbers of features d must be bigger or equal than the number of rows n."
        assert rank < self.dim, "The matrix rank must be smaller that the number of features d."
        U_star = ortho_group.rvs(dim=self.nb_samples)
        D_star = np.zeros((self.nb_samples, self.dim))

        for k in range(1, rank + 1):
            # WARNING: For now we have eigenvalues equal to 1 or to 0.
            D_star[k, k] = 1 #rank - k

        return U_star @ D_star @ V_star


class Client:

    def __init__(self, id: int, dim: int, nb_samples: int, S, plunging_dimension: int) -> None:
        super().__init__()
        self.id = id
        self.dim = dim
        self.nb_samples = nb_samples
        self.plunging_dimension = plunging_dimension
        self.S = S
        self.U = None
        self.V = None

    def loss(self):
        return np.linalg.norm(self.S - self.U @ self.V.T, ord='fro') ** 2 / 2

    def local_grad_wrt_U(self):
        """Gradient of F w.r.t. variable U."""
        return (self.U @ self.V.T - self.S) @ self.V

    def local_grad_wrt_V(self):
        """Gradient of F w.r.t. variable V."""
        return (self.U @ self.V.T - self.S).T @ self.U

    def set_U(self, U):
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U = U

    def set_V(self, V):
        assert V.shape == (self.dim, self.plunging_dimension), \
            f"V has not the correct size on client {self.id}"
        self.V = V
