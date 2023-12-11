"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
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

    def generate_neighborood(self):
        # Generate a random upper triangular matrix with 0s and 1s
        upper_triangular = np.triu(np.random.randint(2, size=(self.nb_clients, self.nb_clients)), k=1)

        # Create the symmetric matrix by concatenating the upper triangular matrix with its transpose
        return upper_triangular + upper_triangular.T + np.eye(self.nb_clients, dtype=int)

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
            D_star[k, k] = 1

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
