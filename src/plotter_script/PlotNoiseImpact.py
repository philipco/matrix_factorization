"""
Created by Constantin Philippenko, 11th December 2023.

X-axis: noise level. Y-axis:  logarithm of the loss F after 1000 local iterations.
Goal: illustrate the impact of noise on the loss.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U

import matplotlib

from src.algo.PowerMethods import LocalPower

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 1000
NB_LOCAL_EPOCHS = 10
NB_CLIENTS = 10

USE_MOMENTUM = False

range_noise = [10**-18, 10**-15, 10**-12, 10**-9, 10**-6, 10**-3, 10**-2]

FONTSIZE = 9


def plot_noise_impact(nb_clients: int, nb_samples: int, dim: int, rank_S: int, latent_dim: int, l1_coef: int, l2_coef: int):
    labels = {"power0": r"$\alpha=0$", "POWER": r"$\alpha=1$",  "LocalPower": 'LocalPower'}

    inits = ["power0", "POWER"]
    algo_name = ["LocalPower"]
    related_work = [LocalPower]

    errors = {name: [] for name in inits + algo_name}
    error_at_optimal_solution = {name: [] for name in inits + algo_name}
    cond = {name: [] for name in inits + algo_name}

    errors_optimal = []

    optim_GD = GD_ON_U

    for e in range_noise:
        print(f"=== noise = {e} ===")
        network = Network(nb_clients, nb_samples, dim, rank_S, latent_dim, noise=e)

        for init in inits:
            kappa = np.inf
            print(f"\t== {init} ==")
            while kappa >= 5:
                algo_GD = optim_GD(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=l1_coef, l2_coef=l2_coef)
                kappa = algo_GD.sigma_max / algo_GD.sigma_min
            errors[init].append(algo_GD.run()[-1])
            cond[init].append(algo_GD.sigma_min / algo_GD.sigma_max)
            error_at_optimal_solution[init].append(algo_GD.compute_exact_solution(l1_coef, l2_coef, 0))
        for optim in related_work:
            kappa = np.inf
            algo = optim(network, NB_EPOCHS // NB_LOCAL_EPOCHS, 0.01, "RANDOM", NB_LOCAL_EPOCHS)
            errors[algo.name()].append(algo.run()[-1])
            cond[algo.name()].append(algo.sigma_min / algo.sigma_max)

        ## Optimal error. ###
        S_stacked = np.concatenate([client.S for client in network.clients])
        _, singular_values, _ = scipy.linalg.svd(S_stacked)

        error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                                 min(S_stacked.shape[0],
                                                                                     S_stacked.shape[1]))])
        errors_optimal.append(error_optimal)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"power0": "-.", "BI_power0": "--", "ORTHO": ":", "POWER": (0, (3, 1, 1, 1)), "LocalPower": "-"}
    init_colors = {"power0": COLORS[0], "BI_power0": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5],
                   "LocalPower": COLORS[6]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    for init in inits:
        axs.plot(np.log10(range_noise), np.log10(error_at_optimal_solution[init]), color=init_colors[init], lw=1)
        axs.plot(np.log10(range_noise), np.log10(errors[init]), color=init_colors[init], linestyle=init_linestyle[init])

    for name in algo_name:
        axs.plot(np.log10(range_noise), np.log10(errors[name]), color=init_colors[name], linestyle="-", marker="^")

    axs.set_xlabel(r"Noise level", fontsize=FONTSIZE)

    axs.plot(np.log10(range_noise), np.log10(errors_optimal), color=COLORS[2], lw=2)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=2, label=labels[init]) for init in inits]
    for init in algo_name:
        init_legend.append(Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=2, label=labels[init], marker="^"))
    if errors_optimal != 0:
        init_legend.append(
            Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, label="Exact solution"))
    init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=2, label="Gradient descent"))



    l2 = axs.legend(handles=init_legend, loc='lower right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)

    # Add a zoomed-in region
    x1, x2, y1, y2 = -15.1, -14.85, -25.8, -24.2  # subregion of the original image
    #  specify the position and size of the inset_axes relative to the parent axes
    inset_position = [0.35, 0, 0.3, 0.3] # [left, bottom, width, height]
    axins = axs.inset_axes(
        inset_position,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    for init in inits:
        axins.plot(np.log10(range_noise), np.log10(error_at_optimal_solution[init]), color=init_colors[init], lw=1)
        axins.plot(np.log10(range_noise), np.log10(errors[init]), color=init_colors[init],
                   linestyle=init_linestyle[init])
    for name in algo_name:
        axins.plot(np.log10(range_noise), np.log10(errors[name]), color=init_colors[name], linestyle="-", marker="^")
    axins.plot(np.log10(range_noise), np.log10(errors_optimal), color=COLORS[2], lw=2)
    # Remove tick labels in the zoom-in box
    axins.set_xticks([])
    axins.set_yticks([])
    axs.indicate_inset_zoom(axins, edgecolor="black")

    title = f"../../pictures/convergence_noise_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo_GD.variable_optimization()}"
    if l1_coef != 0:
        title += f"_lasso{l1_coef}"
    if l2_coef != 0:
        title += f"_ridge{l2_coef}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    plot_noise_impact(NB_CLIENTS, 100, 100, 5, 5, 0, 0)
    plot_noise_impact(NB_CLIENTS, 100, 100, 5, 6, 0, 0)