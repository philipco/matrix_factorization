"""
Created by Constantin Philippenko, 11th December 2023.
"""
import networkx as nx
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from skimage import data

from src.Utilities.PytorchUtilities import get_mnist, get_celeba, get_real_sim, get_w8a


def plot_svd(S, network):
    fig, axs = plt.subplots(1, 1, figsize=(3, 4))
    axs.plot(np.log(np.linalg.svd(S)[1]), lw=2)
    axs.set_ylabel(r"$\log(\sigma_k)$", fontsize=9)
    axs.set_xlabel(r"$k$", fontsize=9)
    title = f"../pictures/svd_{network.dataset_name}_N{network.nb_clients}_d{network.dim}"
    if hasattr(network, "noise"):
        title += f"eps{network.noise}"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


class Network:

    def __init__(self, nb_clients: int, nb_samples: int, dim: int, rank_S: int, plunging_dimension: int,
                 noise: int = 0, missing_value: int = 0, dataset_name: str = "synth", seed=1234):
        super().__init__()
        self.dataset_name = dataset_name
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

            self.nb_clients = len(dataset)
            self.plunging_dimension = plunging_dimension

            self.dim = dataset[0].shape[1]
            # self.mask = self.generate_mask(missing_value)

            self.clients = []
            for c_id in range(self.nb_clients):
                self.clients.append(ClientRealData(c_id, self.dim, dataset[c_id].shape[0], dataset[c_id],
                                                   self.plunging_dimension, missing_value, noise))
            self.W = self.generate_neighborood()

        S = np.concatenate([client.S for client in self.clients])
        print("Start to compute the svd of the dataset.")
        plot_svd(S, self)

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


class Client:

    def __init__(self, id: int, dim: int, nb_samples: int, S_star, plunging_dimension: int, missing_value,
                 noise: int = 0) -> None:
        super().__init__()
        self.id = id
        self.dim = dim
        self.nb_samples = nb_samples
        self.plunging_dimension = plunging_dimension
        self.mask = self.generate_mask(missing_value, nb_samples)
        rotation = ortho_group.rvs(dim=self.dim)
        self.S_star = S_star
        self.S = np.copy(self.S_star)
        if noise != 0:
            self.S += np.random.normal(0, noise, size=(self.nb_samples, self.dim))
        if not self.mask.all():
            self.S *= self.mask
        # self.U_star, self.D_star, self.V_star = U_star, D_star, V_star
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = None, None, None, None, None
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = None, None, None, None, None
        self.Phi_V = None

    def generate_mask(self, missing_value, nb_samples):
        if missing_value == 0:
            return np.ones((nb_samples, self.dim))
        return np.random.choice([0, 1], size=(nb_samples, self.dim), p=[missing_value, 1 - missing_value])

    def regularization_term(self, U, l1_coef, l2_coef, nuc_coef):
        return (l1_coef * np.linalg.norm(U, ord=1) + l2_coef * np.linalg.norm(U, ord="fro")**2 / 2
                + nuc_coef * np.linalg.norm(U, ord="nuc"))

    def loss(self, U, V, l1_coef, l2_coef, nuc_coef):
        # S is already multiplied by the mask.
        return (np.linalg.norm(self.S - self.mask * (U @ V.T), ord='fro') ** 2 / 2
                + self.regularization_term(U, l1_coef, l2_coef, nuc_coef))

    def loss_star(self):
        return np.linalg.norm(self.mask * (self.S_star - self.U @ self.V.T), ord='fro') ** 2 / 2

    def local_grad_wrt_U(self, U, l1_coef, l2_coef, nuc_coef):
        """Gradient of F w.r.t. variable U."""
        nuclear_grad = np.zeros((self.nb_samples, self.plunging_dimension))
        if nuc_coef != 0:
            rank = np.linalg.matrix_rank(U)
            u, s, v = np.linalg.svd(U, full_matrices=False)
            nuclear_grad = u[:,:rank] @ v[:,:rank].T
        if not self.mask.all():
            grad = []
            for i in range(self.nb_samples):
                grad_i = np.zeros(self.plunging_dimension)
                for j in range(self.dim):
                    if self.mask[i,j]:
                        grad_i += (self.S[i,j] - self.U[i] @ self.V[j].T) * self.V[j]
                grad.append(-grad_i)
            return np.array(grad)
        sign_U = 2 * (U >= 0) - 1
        return (U @ self.V.T - self.S) @ self.V + l1_coef * sign_U + l2_coef * U + nuc_coef * nuclear_grad

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
            return np.array(grad) + l1_coef * np.sign(V) + l2_coef * V
        return (self.U @ V.T - self.S).T @ self.U + l1_coef * np.sign(V) + l2_coef * V

    def set_initial_U(self, U):
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = U, U, U, U, U

    def set_U(self, U):
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U = U

    def set_initial_V(self, V):
        assert V.shape == (self.dim, self.plunging_dimension), \
            f"V has not the correct size on client {self.id}"
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = V, V, V, V, V

    def set_V(self, V):
        self.V = V

    def local_power_iteration(self):
        # We update V.
        V = self.S.T @ self.S @ self.V / self.nb_samples

        # We orthogonalize V.
        V = scipy.linalg.orth(V, rcond=0)

        # We compute U.
        U = self.S @ V

        self.set_U(U)
        self.set_V(V)
        return V


class ClientRealData(Client):

    def __init__(self, id: int, dim: int, nb_samples: int, S_star, plunging_dimension: int, missing_value,
                 noise: int = 0) -> None:
        self.id = id
        self.dim = dim
        self.nb_samples = nb_samples
        self.plunging_dimension = plunging_dimension
        self.mask = self.generate_mask(missing_value, nb_samples)
        self.S_star = self.S = S_star.astype(np.float64)
        self.S = np.copy(self.S_star)
        if noise != 0:
            self.S += np.random.normal(0, noise, size=(self.nb_samples, self.dim))
        if not self.mask.all():
            self.S *= self.mask
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = None, None, None, None, None
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = None, None, None, None, None