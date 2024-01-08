"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Client import Network
from src.GradientDescent import GD, AlternateGD, GD_ON_U, GD_ON_V

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 200
NB_CLIENTS = 1
L1_COEF = 0

FONTSIZE=9

if __name__ == '__main__':

    network = Network(NB_CLIENTS, 100, 100, 5, 5, missing_value=0.5)

    optimizations = {"V": GD_ON_V, "U": GD_ON_U}
    errors = {"UV": {}, "V": {}, "U": {}}
    inits = ["SMART", "BI_SMART", "ORTHO"]

    # RANDOM initialization for optimization on U,V
    # algo = AlternateGD(network, NB_EPOCHS, 0.01, "RANDOM")
    # errors["UV"]["RANDOM"] = algo.gradient_descent()
    # print(f"RANDOM\terror min:", errors["UV"]["RANDOM"][-1])

    for key in optimizations.keys():
        print(f"=== {key} ===")
        for init in inits:
            print(f"\t== {init} ==")
            algo = optimizations[key](network, NB_EPOCHS, 0.01, init)
            errors[key][init] = algo.gradient_descent()
            # error_exact = algo.exact_solution()
            print(f"{init}\terror min:", errors[key][init][-1])

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    optim_colors = {"UV": COLORS[0], "V": COLORS[1], "U": COLORS[2]}
    init_linestyle = {"RANDOM": "-.", "SMART": "-", "BI_SMART": "--", "ORTHO": ":"}


    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    # axs.plot(np.log10(errors["UV"]["RANDOM"]), color=optim_colors["UV"], linestyle=init_linestyle["RANDOM"])
    for key in optimizations.keys():
        for init in inits:
            axs.plot(np.log10(errors[key][init]), color=optim_colors[key], linestyle=init_linestyle[init])

    gd_legend = [Line2D([0], [0], color=COLORS[0], lw=2, label='gd on U, V'),
                   Line2D([0], [0], color=COLORS[1], lw=2, label='gd on V'),
                   Line2D([0], [0], color=COLORS[2], lw=2, label='gd on U')]

    init_legend = [Line2D([0], [0], linestyle="-.", color="black", lw=2, label='random'),
                   Line2D([0], [0], color="black", lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color="black", lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color="black", lw=2, label='ortho')]

    l1 = axs.legend(handles=gd_legend, loc='lower left', fontsize=FONTSIZE)
    l2 = axs.legend(handles=init_legend, loc='center right', fontsize=FONTSIZE)

    axs.add_artist(l1)
    axs.add_artist(l2)

    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)

    plt.savefig(f"../pictures/convergence_N{network.nb_clients}_r{network.plunging_dimension}.pdf", dpi=600, bbox_inches='tight')

