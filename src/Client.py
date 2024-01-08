"""
Created by Constantin Philippenko, 11th December 2023.
"""
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ortho_group


class Network:

    def __init__(self, nb_clients: int, nb_samples: int, dim: int, rank_S: int, plunging_dimension: int,
                 noise: int = 0, missing_value: int = 0):
        super().__init__()
        self.nb_clients = nb_clients
        self.dim = dim
        self.plunging_dimension = plunging_dimension
        self.nb_samples = nb_samples
        self.mask = self.generate_mask(missing_value)
        self.clients = self.generate_network_of_clients(rank_S, noise)
        self.W = self.generate_neighborood()
        self.plot_graph_connectivity()

    def generate_mask(self, missing_value):
        if missing_value == 0:
            return np.ones((self.nb_samples, self.dim))
        return np.random.choice([0, 1], size=(self.nb_samples, self.dim), p=[missing_value, 1 - missing_value])

    def plot_graph_connectivity(self):
        if self.nb_clients == 1:
            return
        # We remove the self connection to avoid loop on the graph.
        G = nx.from_numpy_matrix(self.W - np.diag(np.diag(self.W)))
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
            W = nx.to_numpy_matrix(nx.random_regular_graph(4, self.nb_clients, seed=456)) + np.eye(self.nb_clients, dtype=int)
        elif type == "ERDOS":
            W = nx.to_numpy_matrix(nx.fast_gnp_random_graph(self.nb_clients, 0.5, seed=456)) + np.eye(self.nb_clients, dtype=int)
        else:
            raise ValueError("Unrecognized type of connectivity graph.")
        W = np.array(W / np.sum(W, axis=1))
        assert (np.sum(W, axis=1) == 1).all(), "W is not doubly stochastic."
        return W

    def generate_network_of_clients(self, rank_S: int, noise: int = 0):
        np.random.seed(1234)
        clients = []
        V_star = ortho_group.rvs(dim=self.dim)[:rank_S].T
        for c_id in range(self.nb_clients):
            U_star, D_star = self.generate_low_rank_matrix(rank_S)
            clients.append(Client(c_id, self.dim, self.nb_samples, U_star, D_star, V_star, self.plunging_dimension,
                                  self.mask, noise))
        return clients

    def generate_low_rank_matrix(self, rank: int):
        assert self.dim >= self.nb_samples, "The numbers of features d must be bigger or equal than the number of rows n."
        assert rank < self.dim, "The matrix rank must be smaller that the number of features d."
        U_star = ortho_group.rvs(dim=self.nb_samples)[:rank].T
        D_star = np.zeros((rank, rank))

        D_star[0, 0] = 1
        for k in range(1, rank):
            # WARNING: For now we have eigenvalues equal to 1 or to 0.
            D_star[k, k] = 1

        return U_star, D_star

    def reset_eig(self, eigs):
        for client in self.clients:
            client.reset_eig(eigs)


class Client:

    def __init__(self, id: int, dim: int, nb_samples: int, U_star, D_star, V_star, plunging_dimension: int, mask,
                 noise: int = 0) -> None:
        super().__init__()
        self.id = id
        self.dim = dim
        self.nb_samples = nb_samples
        self.plunging_dimension = plunging_dimension
        self.S_star = U_star @ D_star @ V_star.T
        if noise != 0:
            print("Adding some noise.")
            self.S = self.S_star + np.random.normal(0, noise, size=(self.nb_samples, self.dim))
        else:
            self.S = self.S_star
        self.U_star, self.D_star, self.V_star = U_star, D_star, V_star
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = None, None, None, None, None
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = None, None, None, None, None
        self.mask = mask

    def reset_eig(self, eigs):
        for k in range(len(eigs)):
            # WARNING: For now we have eigenvalues equal to 1 or to 0.
            self.D_star[k, k] = eigs[k]
        self.S = self.U_star @ self.D_star @ self.V_star.T

    def loss(self):
        return np.linalg.norm(self.S - self.U @ self.V.T, ord='fro') ** 2 / 2

    def loss_mask(self):
        return np.linalg.norm(self.mask * (self.S - self.U @ self.V.T), ord='fro') ** 2 / 2

    def loss_star(self):
        return np.linalg.norm(self.S_star - self.U @ self.V.T, ord='fro') ** 2 / 2

    def local_grad_wrt_U(self, U, l1_coef, l2_coef):
        """Gradient of F w.r.t. variable U."""
        if not self.mask.all():
            grad = []
            for i in range(self.nb_samples):
                grad_i = np.zeros(self.plunging_dimension)
                for j in range(self.dim):
                    if self.mask[i,j]:
                        grad_i += (self.S[i,j] - self.U[i] @ self.V[j].T) * self.V[j]
                grad.append(-grad_i)
            return np.array(grad) + l1_coef * np.sign(U) + l2_coef * (U - self.U_0)
        return (U @ self.V.T - self.S) @ self.V + l1_coef * np.sign(U) + l2_coef * (U - self.U_0)

    def local_grad_wrt_V(self, V, l1_coef, l2_coef):
        """Gradient of F w.r.t. variable V."""
        # If there is a missing values, then the gradient is more complex.
        if not self.mask.all():
            grad = []
            for j in range(self.dim):
                grad_j = np.zeros(self.plunging_dimension)
                for i in range(self.nb_samples):
                    if self.mask[i, j]:
                        grad_j += (self.S[i, j] - self.U[i] @ self.V[j].T) * self.U[i]
                grad.append(-grad_j)
            return np.array(grad) + l1_coef * np.sign(V) + l2_coef * (V - self.V_0)
        return (self.U @ V.T - self.S).T @ self.U + l1_coef * np.sign(V) + l2_coef * (V - self.V_0)

    def set_U(self, U):
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = U, U, U, U, U

    def set_V(self, V):
        assert V.shape == (self.dim, self.plunging_dimension), \
            f"V has not the correct size on client {self.id}"
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = V, V, V, V, V
