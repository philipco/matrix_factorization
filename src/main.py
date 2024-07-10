"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U

import matplotlib

from src.algo.PowerMethods import DistributedPowerMethod

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 20
NB_CLIENTS = 10
L1_COEF = 0
L2_COEF = 0*10**-6
NUC_COEF = 10**-3

NOISE = 0*10**-6

FONTSIZE=9

if __name__ == '__main__':

    network = Network(NB_CLIENTS, 100, 100, 5, 6, noise=NOISE)

    optimizations = {"U": GD_ON_U}
    errors = {"UV": {}, "V": {}, "U": {}}
    inits = ["SMART", "POWER"]

    # # RANDOM initialization for optimization on U,V
    # algo = DistributedPowerMethod(network, NB_EPOCHS // 10, 0.01, "RANDOM", 10)
    # errors["UV"]["RANDOM"] = algo.run()
    # print(f"{algo.init_type}\terror min:", errors["UV"]["RANDOM"][-1])

    for key in optimizations.keys():
        print(f"=== {key} ===")
        for init in inits:
            print(f"\t== {init} ==")
            algo = optimizations[key](network, NB_EPOCHS, 0.01, init, l1_coef=L1_COEF, l2_coef=L2_COEF,
                                      nuc_coef=NUC_COEF)
            errors[key][init] = algo.run()
            algo.compute_exact_solution(0, 0, 0)
            print(f"{init}\terror min:", errors[key][init][-1])

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    optim_colors = {"UV": COLORS[0], "V": COLORS[1], "U": COLORS[2]}
    init_linestyle = {"RANDOM": "-.", "SMART": "-", "BI_SMART": "--", "ORTHO": ":", "POWER": (0, (3, 1, 1, 1))}


    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(np.log10(errors["UV"]["RANDOM"]), color=optim_colors["UV"], linestyle=init_linestyle["RANDOM"])
    for key in optimizations.keys():
        for init in inits:
            axs.plot(np.log10(errors[key][init]), color=optim_colors[key], linestyle=init_linestyle[init])

    gd_legend = [Line2D([0], [0], color=COLORS[0], lw=2, label='GD on (U, V)'),
                   Line2D([0], [0], color=COLORS[2], lw=2, label='GD on U')]

    init_legend = [Line2D([0], [0], linestyle="-.", color="black", lw=2, label='random'),
                   Line2D([0], [0], color="black", lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color="black", lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color="black", lw=2, label='ortho'),
                   Line2D([0], [0], linestyle=init_linestyle["POWER"], color="black", lw=2, label='power')]

    l1 = axs.legend(handles=gd_legend, loc='upper right', fontsize=FONTSIZE)
    l2 = axs.legend(handles=init_legend, loc='center right', fontsize=FONTSIZE)

    axs.add_artist(l1)
    axs.add_artist(l2)

    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)

    title = f"../pictures/convergence_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if NOISE != 0:
        title += f"_eps{NOISE}"
    if algo.l1_coef != 0:
        title += f"_lasso{algo.l1_coef}"
    if algo.l2_coef != 0:
        title += f"_ridge{algo.l2_coef}"

    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')
