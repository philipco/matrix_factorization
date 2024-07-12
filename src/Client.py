"""
Created by Constantin Philippenko, 11th December 2023.
"""

import numpy as np
import scipy
from scipy.stats import ortho_group


class Client:
    """Create a client of the federated network with its own local dataset."""

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
        """Generate a random mask (to simulate missing values)."""
        if missing_value == 0:
            return np.ones((nb_samples, self.dim))
        return np.random.choice([0, 1], size=(nb_samples, self.dim), p=[missing_value, 1 - missing_value])

    def regularization_term(self, U, l1_coef, l2_coef, nuc_coef):
        """Compute the regularization terms (L1, L2 and nuclear)."""
        return (l1_coef * np.linalg.norm(U, ord=1) + l2_coef * np.linalg.norm(U, ord="fro")**2 / 2
                + nuc_coef * np.linalg.norm(U, ord="nuc"))

    def loss(self, U, V, l1_coef, l2_coef, nuc_coef):
        """Evaluate the local objective function."""
        # S is already multiplied by the mask.
        return (np.linalg.norm(self.S - self.mask * (U @ V.T), ord='fro') ** 2 / 2
                + self.regularization_term(U, l1_coef, l2_coef, nuc_coef))

    def loss_star(self):
        """Evaluate the local objective function but w.r.t. the local true low-rank and unnoised matrix S_*."""
        return np.linalg.norm(self.S_star - self.U @ self.V.T, ord='fro') ** 2 / 2

    def local_grad_wrt_U(self, U, l1_coef, l2_coef, nuc_coef):
        """Compute the local gradient w.r.t. the matrix U."""
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

    def set_initial_U(self, U):
        """Initialize matrix U."""
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U, self.U_0, self.U_avg, self.U_past, self.U_half = U, U, U, U, U

    def set_U(self, U):
        """Set matrix U."""
        assert U.shape == (self.nb_samples, self.plunging_dimension), \
            f"U has not the correct size on client {self.id}"
        self.U = U

    def set_initial_V(self, V):
        """Initialize matrix V."""
        assert V.shape == (self.dim, self.plunging_dimension), \
            f"V has not the correct size on client {self.id}"
        self.V, self.V_0, self.V_avg, self.V_past, self.V_half = V, V, V, V, V

    def set_V(self, V):
        """Set matrix U."""
        self.V = V

    def local_power_iteration(self):
        """Run on the local client a step of power iteration."""
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
    """Create a client of the federated network with its own local dataset. The dataset is a real one and not a
    synthetic one."""
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