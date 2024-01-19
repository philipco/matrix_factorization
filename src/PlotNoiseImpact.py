"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.Client import Network
from src.algo.GradientDescent import GD_ON_U, GD_ON_V

import matplotlib

from src.algo.PowerMethods import DistributedPowerMethod

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

L1_COEF = 0
L2_COEF = 0
range_noise = [10**-15, 10**-12, 10**-9, 10**-6, 10**-3, 10**-2]

FONTSIZE = 9

if __name__ == '__main__':

    # network = Network(NB_CLIENTS, None, None, None, 100, noise=0,
    #                   image_name="cameran")


    inits = ["SMART", "BI_SMART", "ORTHO", "POWER"]
    labels = {"SMART": 'smart', "BI_SMART": 'bismart', "ORTHO": "ortho", "POWER": 'power', "LocalPower": 'LocalPower'}
    algo_name = ["LocalPower"]
    related_work = [DistributedPowerMethod]

    errors = {name: [] for name in inits + algo_name}
    error_at_optimal_solution = {name: [] for name in inits + algo_name}
    cond = {name: [] for name in inits + algo_name}

    error_optimal = []

    optim_GD = GD_ON_U

    vector_values = np.array([])  # To evaluate sparsity.
    for e in range_noise:
        print(f"=== noise = {e} ===")
        network = Network(NB_CLIENTS, 100, 100, 5, 6, noise=e)

        for init in inits:
            kappa = np.inf
            print(f"\t== {init} ==")
            while kappa >= 5:
                algo_GD = optim_GD(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
                kappa = algo_GD.sigma_max / algo_GD.sigma_min
            errors[init].append(algo_GD.run()[-1])
            cond[init].append(algo_GD.sigma_min / algo_GD.sigma_max)
            error_at_optimal_solution[init].append(algo_GD.compute_exact_solution(L1_COEF, L2_COEF))
        for optim in related_work:
            kappa = np.inf
            algo = optim(network, NB_EPOCHS // NB_LOCAL_EPOCHS, 0.01, "RANDOM", NB_LOCAL_EPOCHS)
            errors[algo.name()].append(algo.run()[-1])
            cond[algo.name()].append(algo.sigma_min / algo.sigma_max)
        error_optimal.append(np.mean(
            [np.linalg.norm(client.S - client.S_star, ord='fro') ** 2 / 2 for client in network.clients]))


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "-.", "BI_SMART": "--", "ORTHO": ":", "POWER": (0, (3, 1, 1, 1)), "LocalPower": "-"}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5],
                   "LocalPower": COLORS[6]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    for init in inits:
        axs.plot(np.log10(range_noise), np.log10(error_at_optimal_solution[init]), color=init_colors[init], lw=1,
                 marker="^")
        axs.plot(np.log10(range_noise), np.log10(errors[init]), color=init_colors[init], linestyle=init_linestyle[init])

    for name in algo_name:
        axs.plot(np.log10(range_noise), np.log10(errors[name]), color=init_colors[name], linestyle="-")

    axs.set_xlabel(r"Noise level", fontsize=FONTSIZE)

    axs.plot(np.log10(range_noise), np.log10(error_optimal), color=COLORS[2], lw=2)

    init_legend = [Line2D([0], [0], linestyle=init_linestyle[init], color=init_colors[init],
                          lw=2, label=labels[init]) for init in inits + algo_name]
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, marker="^",
                              label=r'$\| S - \hat{S} \|^2_F$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$\| S - S_* \|^2_F$'))



    l2 = axs.legend(handles=init_legend, loc='lower right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)

    # Add a zoomed-in region
    x1, x2, y1, y2 = -15.05, -14.8, -26.4, -26  # subregion of the original image
    #  specify the position and size of the inset_axes relative to the parent axes
    inset_position = [0.35, 0, 0.3, 0.3] # [left, bottom, width, height]
    axins = axs.inset_axes(
        inset_position,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    for init in inits:
        axins.plot(np.log10(range_noise), np.log10(error_at_optimal_solution[init]), color=init_colors[init], lw=1,
                 marker="^")
        axins.plot(np.log10(range_noise), np.log10(errors[init]), color=init_colors[init],
                   linestyle=init_linestyle[init])
    for name in algo_name:
        axins.plot(np.log10(range_noise), np.log10(errors[name]), color=init_colors[name], linestyle="-")
    axins.plot(np.log10(range_noise), np.log10(error_optimal), color=COLORS[2], lw=2)
    # Remove tick labels in the zoom-in box
    axins.set_xticks([])
    axins.set_yticks([])
    axs.indicate_inset_zoom(axins, edgecolor="black")

    title = f"../pictures/convergence_noise_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo_GD.variable_optimization()}"
    if L1_COEF != 0:
        title += f"_lasso{L1_COEF}"
    if L2_COEF != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


