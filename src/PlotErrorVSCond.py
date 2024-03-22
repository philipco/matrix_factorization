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
NB_CLIENTS = 10
USE_MOMENTUM = False
NB_RUNS = 50

FONTSIZE=9

def plot_errors_vs_condition_number(nb_clients: int, nb_samples: int, dim: int, rank_S: int, latent_dim: int, 
                                    noise: int, l1_coef: int,  l2_coef: int):

    network = Network(nb_clients, nb_samples, dim, rank_S, latent_dim, noise=noise)

    optim = GD_ON_U

    labels = {"SMART": r"$\alpha=1$", "POWER": r"$\alpha=3$"}

    inits = ["SMART", "POWER"]
    errors = {name: [] for name in inits}
    error_at_optimal_solution = {name: [] for name in inits}
    cond = {name: [] for name in inits}
    sigma_min = {name: [] for name in inits}

    momentum = [True, False]

    for init in inits:
        print(f"=== {init} ===")

        vector_values = np.array([]) # To evaluate sparsity.
        for k in range(NB_RUNS):

            algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=l1_coef, l2_coef=l2_coef)
            errors[init].append(algo.run()[-1])
            sigma_min[init].append(algo.sigma_min)
            cond[init].append(algo.sigma_min/algo.sigma_max)
            error_at_optimal_solution[init].append(algo.compute_exact_solution(l1_coef, l2_coef))

            if optim == GD_ON_U:
                vector_values = np.concatenate([vector_values, np.concatenate(network.clients[0].U)])
            elif optim == GD_ON_V:
                vector_values = np.concatenate([vector_values, np.concatenate(network.clients[0].V)])


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "--", "BI_SMART": "--", "ORTHO": ":", "POWER": "--"} #(0, (3, 1, 1, 1))}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x, y = zip(*sorted(zip(cond[init], np.log10(error_at_optimal_solution[init]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=init_colors[init], lw=1)
        else:
            axs.plot(np.array(x) ** 2, y, color=init_colors[init], lw=1)
        x, y = zip(*sorted(zip(cond[init], np.log10(errors[init]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=init_colors[init], linestyle=init_linestyle[init])
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V}) / \sigma_{\mathrm{max}}(\mathbf{V})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=init_colors[init], linestyle=init_linestyle[init])
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V}) / \sigma^2_{\mathrm{max}}(\mathbf{V})$",
                           fontsize=FONTSIZE)

    ## Optimal error. ###
    S_stacked = np.concatenate([client.S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                             min(nb_clients * network.nb_samples,
                                                                                 network.dim))])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in cond["SMART"]]
        if USE_MOMENTUM:
            axs.plot(np.array(cond["SMART"]) ** 1, z, color=COLORS[2], lw=2)
        else:
            axs.plot(np.array(cond["SMART"]) ** 2, z, color=COLORS[2], lw=2)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=2, label=labels[init]) for init in inits]
    if error_optimal != 0:
        init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, label="Exact solution"))
    init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=2, label="Gradient descent"))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_vs_cond_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if noise != 0:
        title += f"_eps{noise}"
    if algo.l1_coef != 0:
        title += f"_lasso{l1_coef}"
    if algo.l2_coef != 0:
        title += f"_ridge{l2_coef}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


    # plt.figure(figsize=(6, 4))
    # vector_values = np.abs(vector_values)
    # vector_values[vector_values < 10**-10] = 10**-12
    # plt.hist(np.log(vector_values), bins=15, alpha=0.7)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # title = f"../pictures/hist_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    # if noise != 0:
    #     title += f"_eps{noise}"
    # if algo.l1_coef != 0:
    #     title += f"_lasso{l1_coef}"
    # if algo.l2_coef != 0:
    #     title += f"_ridge{l2_coef}"
    # if USE_MOMENTUM:
    #     title += f"_momentum"
    # plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')

if __name__ == '__main__':

    # Without noise.
    plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 5, 10**-15, 0,
                                    0)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 0, 0,
    #                                 10**-9)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 0, 0,
    #                                 10 ** -6)
    #
    # # With noise.
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 10 ** -6, 0,
    #                                 0)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 10 ** -6, 0,
    #                                 10 ** -8)
    # plot_errors_vs_condition_number(NB_CLIENTS, 100, 100, 5, 6, 10 ** -6, 0,
    #                                 10 ** -4)