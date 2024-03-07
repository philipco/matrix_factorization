"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Client import Network
from src.algo.GradientDescent import GD_ON_U, GD_ON_V

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 1000
NB_CLIENTS = 1
USE_MOMENTUM = False
NB_RUNS = 100

FONTSIZE=9


def compute_probabilistic_bound_Pii(network, P, r, i, singular_values, sigma_max, eta):
    ### i is the index-1.

    Ustar, singular_values, Vstar = scipy.linalg.svd(network.clients[0].S)
    Vstar = Vstar.T
    Sigma = np.zeros((network.nb_samples, network.dim))
    for j in range(min(network.nb_samples, network.dim)):
        Sigma[j, j] = singular_values[j]

    Phi_hat = Ustar.T @ network.clients[0].Phi

    _, sv_Phi_hat, _ = scipy.linalg.svd(Phi_hat[:network.plunging_dimension].T @ Phi_hat[:network.plunging_dimension])


    ratio_sv = Sigma[network.plunging_dimension-1][network.plunging_dimension-1] ** 2 / Sigma[i][i] ** 2
    numerator = sv_Phi_hat[-1]
    denominator = np.linalg.norm(Phi_hat[i]) ** 2

    probabilistic_bound_Pii = (4 * np.log(eta**-1) * r + 2 * np.log(2) * r**2) / (ratio_sv * eta**2)
    # assert P[i][i] <= probabilistic_bound_Pii or np.isclose(P[i][i], probabilistic_bound_Pii), f"The bound on Pii is not exact for i={i}"
    return probabilistic_bound_Pii


def compute_bound_Pii(network, P, r, i, singular_values, sigma_max, eta):
    ### i is the index-1.

    Ustar, singular_values, Vstar = scipy.linalg.svd(network.clients[0].S)
    Vstar = Vstar.T
    Sigma = np.zeros((network.nb_samples, network.dim))
    for j in range(min(network.nb_samples, network.dim)):
        Sigma[j, j] = singular_values[j]

    Phi_hat = Ustar.T @ network.clients[0].Phi

    _, sv_Phi_hat, _ = scipy.linalg.svd(Phi_hat[:network.plunging_dimension].T @ Phi_hat[:network.plunging_dimension])

    # Phi_hat[:1] is a row matrix.
    eig_Phi_hat_star, _ = np.linalg.eigh(Phi_hat[:network.rank_S].T @ Phi_hat[:network.rank_S])

    ratio_sv = Sigma[network.plunging_dimension-1][network.plunging_dimension-1] ** 2 / Sigma[i][i] ** 2
    ratio_sv_star = Sigma[network.rank_S][network.rank_S-1] ** 2 / Sigma[i][i] ** 2
    numerator = sv_Phi_hat[-1]
    numerator_star = eig_Phi_hat_star[-1]
    denominator = np.linalg.norm(Phi_hat[i]) ** 2

    bound_Pii_star = 1 / (1 + ratio_sv_star * numerator_star / denominator)
    bound_Pii = denominator / (ratio_sv * numerator)
    Pii =  P[i][i]
    assert P[i][i] <= bound_Pii or np.isclose(P[i][i], bound_Pii), f"The bound on Pii is not exact for i={i}"
    return bound_Pii


def compute_theoretical_bounds(network, eta):
    client = network.clients[0]

    r = client.plunging_dimension
    Ustar, singular_values, _ = scipy.linalg.svd(network.clients[0].S)
    sigma_max = np.max(singular_values)
    Sigma = np.zeros((network.nb_samples, network.dim))
    for j in range(min(network.nb_samples, network.dim)):
        Sigma[j, j] = singular_values[j]

    Phi_tilde = Sigma.T @ Ustar.T @ network.clients[0].Phi
    P = Phi_tilde @ np.linalg.pinv(Phi_tilde.T @ Phi_tilde) @ Phi_tilde.T

    upper_bound, upper_probabilistic_bound, almost_exact_bound = 0, 0, 0
    for i in range(network.plunging_dimension+1, min(client.nb_samples, client.dim)):
        upper_Pii = compute_bound_Pii(network, P, r, i, singular_values, sigma_max, eta)
        upper_probabilistic_Pii = compute_probabilistic_bound_Pii(network, P, r, i, singular_values, sigma_max, eta)
        upper_bound += (singular_values[i]**2 + (sigma_max**2 - singular_values[i]**2) * upper_Pii)
        upper_probabilistic_bound += (singular_values[i]**2 + (sigma_max**2 - singular_values[i]**2) * upper_probabilistic_Pii)
        almost_exact_bound += (singular_values[i]**2 + (sigma_max**2 - singular_values[i]**2) * P[i][i])
    assert almost_exact_bound <= upper_bound or np.isclose(almost_exact_bound, upper_bound), "The upper bound on Pii is not correct."
    return upper_bound, upper_probabilistic_bound




def plot_errors_vs_condition_number(nb_clients: int, nb_samples: int, dim: int, rank_S: int, latent_dim: int,
                                    noise: int, l1_coef: int,  l2_coef: int):

    optim = GD_ON_U

    labels = {"SMART": 'smart'}

    inits = ["SMART"]
    error_at_optimal_solution = {name: [] for name in inits}
    theoretical_bounds = {name: [] for name in inits}
    theoretical_probabilistic_bounds = {name: [] for name in inits}

    for init in inits:
        print(f"=== {init} ===")
        for k in range(NB_RUNS):
            network = Network(nb_clients, nb_samples, dim, rank_S, latent_dim, noise=noise, seed=k)

            algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=l1_coef, l2_coef=l2_coef)

            upper_boud, probabilistic_upper_bound = compute_theoretical_bounds(network, 1)
            theoretical_bounds[init].append(upper_boud)
            theoretical_probabilistic_bounds[init].append(probabilistic_upper_bound)


            error_at_optimal_solution[init].append(algo.compute_exact_solution(l1_coef, l2_coef))

            S = np.copy(network.clients[0].S)
            U, singular_values, Vt = scipy.linalg.svd(S)

            Sigma = np.zeros((network.nb_samples, network.dim))

            for i in range(min(network.nb_samples, network.dim)):
                Sigma[i, i] = singular_values[i]

            Phi = Sigma.T @ U.T @ network.clients[0].Phi
            projecteur = Phi @ np.linalg.pinv(Phi.T @ Phi) @ Phi.T
            a = np.linalg.norm(Sigma - Sigma @ projecteur, ord='fro') ** 2

            Phi2 = network.clients[0].S.T @ network.clients[0].Phi
            projecteur2 = Phi2 @ np.linalg.pinv(Phi2.T @ Phi2) @ Phi2.T
            a2 = np.linalg.norm(network.clients[0].S - network.clients[0].U @ network.clients[0].V.T, ord='fro') ** 2
            print("Diff: ", a - a2)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "-.", "BI_SMART": "--", "ORTHO": ":", "POWER": (0, (3, 1, 1, 1))}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x = np.log10(error_at_optimal_solution[init])
        plt.hist(x, bins=30, alpha=0.5, label=init, color="tab:blue")
        plt.hist(np.log10(theoretical_bounds[init]), bins=30, alpha=0.5, label=init, color="tab:orange")
        plt.axvline(np.log10(np.mean(theoretical_probabilistic_bounds[init])), linewidth=2, color="tab:red")
        # plt.axvline(np.log10(theoretical_bounds[init])[0], color="tab:red", linewidth=2)

    ### Optimal error. ###
    singular_values = np.linalg.svd(network.clients[0].S)[1]
    error_optimal = np.sum([singular_values[i]**2 for i in range(network.plunging_dimension+1, min(network.nb_samples, network.dim))])
    z = np.log10(error_optimal)
    # Add a vertical line at the optimal error value
    plt.axvline(x=z, color='tab:green', linewidth=2)

    init_legend = []
    init_legend.append(Line2D([0], [0], linestyle="-", color='tab:blue', lw=2, label=r'$\|\Sigma - \Sigma P \|^2_F$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='tab:orange', lw=2, label=r'$(1 + \frac{\sigma_r^2 }{\sigma_i^2} \frac{\sigma_{\min}^2(\hat{\Phi}_{\leq r})}{ \|\hat{\Phi}_i\|^2_2 })^{-1}$'))

    if error_optimal != 0:
        init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2$'))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/distribution_error_bounds_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if noise != 0:
        title += f"_eps{noise}"
    if algo.l1_coef != 0:
        title += f"_lasso{l1_coef}"
    if algo.l2_coef != 0:
        title += f"_ridge{l2_coef}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')




if __name__ == '__main__':

    # Without noise.
    plot_errors_vs_condition_number(NB_CLIENTS, 50, 100, 5, 5, 10**-5, 0,
                                    0)