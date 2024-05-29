"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Network import Network
from src.MatrixUtilities import power
from src.algo.GradientDescent import GD_ON_U, GD_ON_V

import matplotlib

from src.algo.MFInitialization import generate_gaussian_matrix

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
NB_RUNS = 50

FONTSIZE=9


def compute_probabilistic_bound_Pii(network, r, i, eta, Ustar, Vstar, Sigma, Phi_stacked):

    ratio_sv = Sigma[i][i] ** (2 * (network.power - 1)) / Sigma[r-1][r-1] ** (2 * (network.power - 1))

    probabilistic_bound_Pii = 32 * np.log(4) * r * (r+1) * ratio_sv
    return probabilistic_bound_Pii


def compute_bound_Pii(network, P, r, i, Ustar, Vstar, Sigma, Phi_stacked):
    ### i is the index-1.

    Phi_hat = Ustar.T @ Phi_stacked

    sv_Phi_hat, _ = scipy.linalg.eigh(Phi_hat[:r].T @ Phi_hat[:r])

    ratio_sv = Sigma[i][i] ** (2 * (network.power - 1)) / Sigma[r-1][r-1] ** (2 * (network.power - 1))
    numerator = sv_Phi_hat[0]
    denominator = np.linalg.norm(Phi_hat[i]) ** 2
    bound_Pii = denominator * ratio_sv / numerator
    assert P[i][i] <= bound_Pii or np.isclose(P[i][i], bound_Pii), f"The bound on Pii is not exact for i={i}"
    return P[i][i]


def compute_theoretical_bounds(network, eta):

    r = network.rank_S
    S_stacked = np.concatenate([client.S for client in network.clients])
    Phi_stacked = np.concatenate([generate_gaussian_matrix(client.nb_samples, r,1) for client in network.clients])

    Ustar, singular_values, Vstar = scipy.linalg.svd(S_stacked)
    Vstar = Vstar.T
    sigma_max = np.max(singular_values)
    Sigma = np.zeros((np.sum([c.nb_samples for c in network.clients]), network.dim))
    for j in range(min(np.sum([c.nb_samples for c in network.clients]), network.dim)):
        Sigma[j, j] = singular_values[j]

    Phi_tilde = power(Sigma, network.power) @ Vstar.T @ Phi_stacked
    P = Phi_tilde @ np.linalg.pinv(Phi_tilde.T @ Phi_tilde) @ Phi_tilde.T

    upper_bound, upper_probabilistic_bound, almost_exact_bound = 0, 0, 0
    for i in range(r+1, min(np.sum([c.nb_samples for c in network.clients]), network.dim)):
        upper_Pii = compute_bound_Pii(network, P, r, i, Ustar, Vstar, Sigma, Phi_stacked)
        upper_probabilistic_Pii = compute_probabilistic_bound_Pii(network, r, i, eta, Ustar, Vstar, Sigma, Phi_stacked)
        upper_bound += (singular_values[i]**2 * (1 + (sigma_max**2 - singular_values[i]**2) * upper_Pii / singular_values[r-1]**2))
        upper_probabilistic_bound += (singular_values[i]**2 * (1 + (sigma_max**2 - singular_values[i]**2) * upper_probabilistic_Pii / singular_values[r-1]**2))
        almost_exact_bound += (singular_values[i]**2 + (sigma_max**2 - singular_values[i]**2) * P[i][i])
    assert almost_exact_bound <= upper_bound or np.isclose(almost_exact_bound, upper_bound), "The upper bound on Pii is not correct."
    return upper_bound / 2, upper_probabilistic_bound / 2




def plot_errors_vs_condition_number(nb_clients: int, nb_samples: int, dim: int, rank_S: int, latent_dim: int,
                                    noise: int, l1_coef: int,  l2_coef: int):

    print("= New run. =")

    optim = GD_ON_U

    inits = ["SMART"]
    error_at_optimal_solution = {name: [] for name in inits}
    theoretical_bounds = {name: [] for name in inits}
    theoretical_probabilistic_bounds = {name: [] for name in inits}

    for init in inits:
        print(f"=== {init} ===")
        for k in range(NB_RUNS):
            network = Network(nb_clients, nb_samples, dim, rank_S, latent_dim, noise=noise, seed=k)

            algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=l1_coef, l2_coef=l2_coef)

            upper_boud, probabilistic_upper_bound = compute_theoretical_bounds(network, 0.5)
            theoretical_bounds[init].append(upper_boud)
            theoretical_probabilistic_bounds[init].append(probabilistic_upper_bound)

            error_at_optimal_solution[init].append(algo.compute_exact_solution(l1_coef, l2_coef))


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "-.", "BI_SMART": "--", "ORTHO": ":", "POWER": (0, (3, 1, 1, 1))}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x = np.log10(error_at_optimal_solution[init])
        plt.hist(x, bins=30, alpha=0.5, label=init, color="tab:blue", align="right")
        plt.hist(np.log10(theoretical_bounds[init]), bins=30, alpha=0.5, label=init, color="tab:orange")
        plt.axvline(np.log10(np.mean(theoretical_probabilistic_bounds[init])), linewidth=2, color="tab:red")

    ## Optimal error. ###
    S_stacked = np.concatenate([client.S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                         min(nb_clients * network.nb_samples,
                                                                             network.dim))])
    z = np.log10(error_optimal)
    # Add a vertical line at the optimal error value
    plt.axvline(x=z, color='tab:green', linewidth=2)

    init_legend = []
    init_legend.append(Line2D([0], [0], linestyle="-", color='tab:blue', lw=2, label=r'$\|\mathbf{S} - \mathbf{U} \mathbf{V}^\top \|^2_F / 2$'))
    init_legend.append(
        Line2D([0], [0], linestyle="-", color='tab:red', lw=2, label=r'$\mathbb{P}(\|\mathbf{S} - \mathbf{U} \mathbf{V}^\top\|^2_F / 2 < \delta^{r_*}_{1/2}) \geq 1/2$'))
    # init_legend.append(Line2D([0], [0], linestyle="-", color='tab:orange', lw=2, label=r'$(1 + \frac{\sigma_r^2 }{\sigma_i^2} \frac{\sigma_{\min}^2(\hat{\Phi}_{\leq r})}{ \|\hat{\Phi}_i\|^2_2 })^{-1} / 2$'))

    if error_optimal != 0:
        init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_xlabel("Log(Relative error)", fontsize=FONTSIZE)
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
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 5, 0, 0,
    #                                 0)
    plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 10**-6, 0,
                                    0)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 5, 10 ** -6, 0,
    #                                 0)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 10**-6, 0,
    #                                 0)
    # plot_errors_vs_condition_number(10, 50, 200, 5, 6, 0, 0,
    #                                 0)
    # plot_errors_vs_condition_number(10, 50, 200, 5, 10, 0, 0,
    #                                 0)
    # plot_errors_vs_condition_number(10, 50, 200, 5, 5, 10**-6, 0,
    #                                 0)
    # plot_errors_vs_condition_number(10, 50, 200, 5, 6, 10 ** -6, 0,
    #                                 0)