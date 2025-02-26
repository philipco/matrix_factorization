"""
Created by Constantin Philippenko, 11th December 2023.

X-axis: condition number. Y-axis:  logarithm of the loss F after 1000 local iterations.
Goal: illustrate the impact of the sampled Gaussian matrices Phi on the convergence rate.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U

import matplotlib

from src.utilities.data.DatasetsSettings import *

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

USE_MOMENTUM = False
NB_RUNS = 50
DATASET_NAME = "synth"
L1_COEF = 0 #10**-2
L2_COEF = 0 #10**-3
NUC_COEF = 0

FONTSIZE=24
LW=2


def plot_errors_vs_condition_number(noise:int, l1_coef: int,  l2_coef: int,  nuc_coef: int):

    network = Network(NB_CLIENTS[DATASET_NAME], 200, 200, RANK_S[DATASET_NAME],
                      LATENT_DIMENSION[DATASET_NAME], noise=noise, dataset_name=DATASET_NAME)

    optim = GD_ON_U

    labels = {"power0": r"$\alpha=0$", "power1": r"$\alpha=1$"}

    inits = ["power0", "power1"]
    errors = {name: [] for name in inits}
    error_at_optimal_solution = {name: [] for name in inits}
    cond = {name: [] for name in inits}
    sigma_min = {name: [] for name in inits}

    for init in inits:
        print(f"=== {init} ===")

        vector_values = np.array([]) # To evaluate sparsity.
        for k in range(NB_RUNS):

            algo = optim(network, 1000, init, L1_COEF, L2_COEF, NUC_COEF,
                                use_momentum=USE_MOMENTUM)
            output, nb_it = algo.run()
            errors[init].append(output[-1])
            sigma_min[init].append(algo.sigma_min)
            cond[init].append(algo.sigma_min/algo.sigma_max)
            # All Nuclear and L1 coefficients are set to zero when computing the exact solution.
            error_at_optimal_solution[init].append(algo.compute_exact_solution(l1_coef, l2_coef, nuc_coef))

            vector_values = np.concatenate([vector_values, np.concatenate(network.clients[0].U)])


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"power0": "--", "power1": "--"}
    init_colors = {"power0": COLORS[0], "power1": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 2.25))
    for init in inits:
        x, y = zip(*sorted(zip(cond[init], np.log10(error_at_optimal_solution[init]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=init_colors[init], lw=LW)
        else:
            axs.plot(np.array(x) ** 2, y, color=init_colors[init], lw=LW)

        # Plot error after our GD implementation.
        x, y = zip(*sorted(zip(cond[init], np.log10(errors[init]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=init_colors[init], linestyle=init_linestyle[init], lw=LW)
            # axs.set_xlabel(r"$\kappa^{-1}(\mathbf{V})$",
            #                fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=init_colors[init], linestyle=init_linestyle[init], lw=LW)
            # axs.set_xlabel(r"$\kappa^{-2}(\mathbf{V})$",
            #                fontsize=FONTSIZE)

    ## Optimal error. ###
    S_stacked = np.concatenate([client.S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                         min(np.sum([c.nb_samples for c in network.clients]),
                                                                             network.dim))])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in cond["power0"]]
        if USE_MOMENTUM:
            axs.plot(np.array(cond["power0"]) ** 1, z, color=COLORS[2], lw=LW)
        else:
            axs.plot(np.array(cond["power0"]) ** 2, z, color=COLORS[2], lw=LW)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=LW, label=labels[init]) for init in inits]
    if error_optimal != 0:
        init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=LW,
                                  label=r'$ \sum_{i>r} \frac{\sigma_i^2}{2}$'))
    # init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=LW, label="Exact solution"))
    # init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=LW, label="Gradient descent"))

    if noise == 0:
        l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE, borderaxespad=0.1, labelspacing=0,
                        handletextpad=0.2)

        axs.add_artist(l2)
    # axs.set_ylabel(r"$\log_{10}(\|\mathbf{S} - \mathbf{U} \mathbf{V}^\top \|_{\mathrm{F}})$", fontsize=FONTSIZE)
    title = f"../../pictures/convergence_vs_cond_{DATASET_NAME}_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if NOISE[DATASET_NAME] != 0:
        title += f"_eps{noise}"
    if algo.l1_coef != 0:
        title += f"_lasso{l1_coef}"
    if algo.l2_coef != 0:
        title += f"_ridge{l2_coef}"
    if algo.nuc_coef != 0:
        title += f"nuc{l2_coef}"
    if USE_MOMENTUM:
        title += f"_momentum"

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


if __name__ == '__main__':

    # Without noise.
    plot_errors_vs_condition_number(0, 0, 0, 0)
    plot_errors_vs_condition_number(NOISE["synth"], 0, 0, 0)