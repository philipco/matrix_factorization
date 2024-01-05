import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank
from scipy.stats import ortho_group

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

FONTSIZE = 15

def generate_gaussian_matrix(n, d, std=1):
    gaussian_matrix = np.random.normal(0, std, size=(n, d))
    return gaussian_matrix

def generate_low_rank_matrix(n, d, V_star, rank: int):
    assert d <= n, "The numbers of features d must be bigger or equal than the number of rows n."
    assert rank < d, "The matrix rank must be smaller that the number of features d."
    U_star = ortho_group.rvs(dim=n)
    D_star = np.zeros((n, d))

    for k in range(1, rank+1):
        D_star[k,k] = 1

    return U_star @ D_star @ V_star

# Interesting : n = d = 50, rand(S*)=5, std = 0.001, r = 6, nu = 0.5, eig1 = 1, N = 1 ne converge pas, N = 10 converge.

nb_clients = 1

# Replace these values with your desired dimensions
d = 100  # number of columns
n = [100 for i in range(nb_clients)] #np.random.randint(d, 2*d, nb_clients)    # random number of rows, between d and
V_star = ortho_group.rvs(dim=d)


rank = 5
low_rank = 6
C = 4
nu = 2
D = C * nu / 9
bigest_sv = 1

std_error = 10**(-4)

E_star = [generate_gaussian_matrix(n[i], d, std_error) for i in range(nb_clients)]
S_star = [generate_low_rank_matrix(n[i] , d, V_star, rank) for i in range(nb_clients)]
S = [s for (s, e) in zip(S_star, E_star)]

U, diag, V = np.linalg.svd(S[0])

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
axs.plot(np.log(np.sort(np.diag(diag))), color="tab:blue")
axs.set_xlabel("$i \in [\min(n, d)]$", fontsize=FONTSIZE)
axs.set_ylabel("$(\log(\mathrm{eig}(\mathbf{S})))_{i \in [\min(n, d)]}$", fontsize=FONTSIZE)
plt.savefig("eigenvalues.png", dpi=100)

print("Rank of E*(0):", matrix_rank(E_star[0]))
print("Rank of S*(0):", matrix_rank(S_star[0]))
print("Rank of S(0):", matrix_rank(S[0]))


def generate_neighborood(n: int):
    # Generate a random upper triangular matrix with 0s and 1s
    upper_triangular = np.triu(np.random.randint(2, size=(n, n)), k=1)

    # Create the symmetric matrix by concatenating the upper triangular matrix with its transpose
    return upper_triangular + upper_triangular.T + np.eye(n, dtype=int)

W = generate_neighborood(nb_clients)
print("Graph of connections:")
print(W)

# Example usage:
def F(S, U, V):
    # Define your objective function here
    return np.mean([np.linalg.norm(s - u @ v.T, ord = 'fro')**2 / 2 for (s,u, v) in zip(S, U, V)])


def local_grad_F_wrt_U(S, U, V, nb_clients: int, E = None):
    """Gradient of F w.r.t. variable U."""
    if E is None:
        return (U @ V.T - S) @ V / nb_clients
    return (U @ V.T + E - S) @ V / nb_clients


def local_grad_F_wrt_V(S, U, V, nb_clients: int, E = None):
    """Gradient of F w.r.t. variable V."""
    if E is None:
        return (U @ V.T - S).T @ U / nb_clients
    return (U @ V.T + E - S).T @ U / nb_clients


def local_grad_F_wrt_E(S, U, V, nb_clients: int, E):
    """Gradient of F w.r.t. variable E."""
    return (U @ V.T + E - S) / nb_clients


def smart_init(S: np.ndarray, low_rank: int, step_size: int) -> [np.ndarray, np.ndarray]:
    d = S[0].shape[1]
    Phi, Phi_prime = generate_gaussian_matrix(d, low_rank, 1), generate_gaussian_matrix(d, low_rank, 1)
    U_0 = [s @ Phi / (np.sqrt(step_size * low_rank) * C * bigest_sv) for s in S]
    V_0 = [Phi_prime * np.sqrt(step_size) * D * bigest_sv / np.sqrt(d) for s in S]
    return U_0, V_0


def gradient_descent(S: np.ndarray, low_rank: int, step_size, num_iterations, with_smart_init: bool = True, decentralized = False, with_error = False):
    nb_clients = len(S)
    d = S[0].shape[1]
    if with_smart_init:
        U0, V0 = smart_init(S, low_rank, step_size)
    else:
        U0, V0 = ([generate_gaussian_matrix(n[i], low_rank, 1) / (10 * np.sqrt(low_rank)) for i in range(nb_clients)],
                  [generate_gaussian_matrix(d, low_rank, 1) / (10 * np.sqrt(d)) for i in range(nb_clients)])
    if with_error:
        E0 = [generate_gaussian_matrix(n, d, 1) for i in range(nb_clients)]
        E = E0.copy()
    else:
        E = [None for i in range(nb_clients)]
    U, V = U0.copy(), V0.copy()
    error_S = [F(S, U0, V0)]
    error_S_star = [F(S_star, U0, V0)]

    for i in range(num_iterations):
        Vold = V.copy()
        for j in range(nb_clients):
            U[j] -= (step_size * local_grad_F_wrt_U(S[j], U[j], V[j], nb_clients, E[j]))
            if not decentralized:
                V[j] -= (step_size * np.sum([local_grad_F_wrt_V(S[k], U[k], V[j], nb_clients, E[j]) for k in range(nb_clients)], axis=0))
            else:
                rho = 0.001
                V[j] = (1-rho) * Vold[j] + rho * np.mean([ W[j-1,k-1] * Vold[k] for k in range(nb_clients)], axis=0) - (step_size * local_grad_F_wrt_V(S[j], U[j], V[j], nb_clients, E[j]))
        if E[0] is not None:
            for j in range(nb_clients):
                E[j] -= (step_size * np.sum([local_grad_F_wrt_E(S[i], U[i], V[j], nb_clients, E[j]) for i in range(nb_clients)], axis=0))
        error_S.append(F(S, U, V))
        error_S_star.append(F(S_star, U, V))

    return [U, V, E], error_S, error_S_star


step_size = 9 / (4 * C * nu * bigest_sv)
print("step_size", step_size)
num_iterations = 150

# Run gradient descent
result_params, error_S, error_S_star = gradient_descent(S, low_rank, step_size, num_iterations)
result_params_wo, error_S_wo, error_S_star_wo = gradient_descent(S, low_rank, step_size, num_iterations, with_smart_init=False)

# print("Optimal parameters:", result_params)
print("Minimum value of F:", F(S, result_params[0], result_params[1]))

# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# axs[0].plot(np.log10(error_S), label="w. smart init.")
# axs[0].plot(np.log10(error_S_wo), label="w.o. smart init.")
# axs[0].legend()
# axs[0].set_title("Error to $\mathbf{S}$")
# axs[1].plot(np.log10(error_S_star), label="w. smart init.")
# axs[1].plot(np.log10(error_S_star_wo), label="w.o. smart init.")
# axs[1].set_title("Error to $\mathbf{S}_*$")
# plt.show()

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
axs.plot(np.log10(error_S_star), label="w. smart init.")
axs.plot(np.log10(error_S_star_wo), label="w.o. smart init.")
axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
# axs.set_ylabel("$\\frac{1}{2N} \sum_{i=1}^N \|\mathbf{S} - \mathbf{U} \mathbf{V}^\\top \|^2_ F$", fontsize=FONTSIZE)
axs.set_ylabel("Relative error", fontsize=FONTSIZE)
# axs.set_title("Error to $\mathbf{S}$", fontsize=FONTSIZE)
axs.legend(fontsize=FONTSIZE)
plt.savefig(f"convergence_N{nb_clients}.png", dpi=100)



