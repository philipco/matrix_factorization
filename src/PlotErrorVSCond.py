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

NB_EPOCHS = 2000
NB_CLIENTS = 1

NB_RUNS = 50

FONTSIZE=9

if __name__ == '__main__':


    network = Network(NB_CLIENTS, 100, 100, 5, 5)

    # optimizations = {"UV": AlternateGD, "V": GD_ON_V, "U": GD_ON_U}
    optim = GD_ON_U
    errors = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    sigma_min = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    cond = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    inits = ["SMART", "BI_SMART", "ORTHO"]

    momentum = [True, False]

    for init in inits:
        print(f"=== {init} ===")
        for k in range(NB_RUNS):
            algo = optim(network, NB_EPOCHS, 0.01, init)
            errors[init].append(algo.gradient_descent()[-1])
            sigma_min[init].append(algo.sigma_min)
            cond[init].append(algo.sigma_min/algo.sigma_max)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    optim_colors = {"UV": COLORS[0], "V": COLORS[1], "U": COLORS[2]}
    init_linestyle = {"RANDOM": "-.", "SMART": "-", "BI_SMART": "--", "ORTHO": ":"}
    init_colors = {"RANDOM": COLORS[0], "SMART": COLORS[1], "BI_SMART": COLORS[2], "ORTHO": COLORS[3]}


    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x, y = zip(*sorted(zip(sigma_min[init], np.log10(errors[init]))))
        axs.plot(np.array(x) ** 2, y, color=init_colors[init], linestyle=init_linestyle[init])

    init_legend = [Line2D([0], [0], linestyle="-", color=COLORS[1], lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color=COLORS[2], lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color=COLORS[3], lw=2, label='ortho')]

    l2 = axs.legend(handles=init_legend, loc='center right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0})$", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)
    plt.savefig(f"convergence_vs_sigma_N{network.nb_clients}_r{network.plunging_dimension}.pdf", dpi=600, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x, y = zip(*sorted(zip(cond[init], np.log10(errors[init]))))
        axs.plot(np.array(x) ** 1, y, color=init_colors[init], linestyle=init_linestyle[init])

    init_legend = [Line2D([0], [0], linestyle="-", color=COLORS[1], lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color=COLORS[2], lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color=COLORS[3], lw=2, label='ortho')]

    l2 = axs.legend(handles=init_legend, loc='center right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$", fontsize=FONTSIZE)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    plt.savefig(f"convergence_vs_cond_N{network.nb_clients}_r{network.plunging_dimension}.pdf", dpi=600,
                bbox_inches='tight')

