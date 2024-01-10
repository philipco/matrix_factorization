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

NB_EPOCHS = 400
NB_CLIENTS = 1

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0

NB_RUNS = 30

FONTSIZE=9

if __name__ == '__main__':


    network = Network(NB_CLIENTS, 100, 100, 5, 6, noise=10**-5)

    optim = GD_ON_U
    errors = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    sigma_min = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    cond = {"RANDOM": [], "SMART": [], "BI_SMART": [], "ORTHO": []}
    inits = ["SMART", "BI_SMART", "ORTHO"]

    momentum = [True, False]

    for init in inits:
        print(f"=== {init} ===")
        
        vector_values = np.array([]) # To evaluate sparsity.
        for k in range(NB_RUNS):

            algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
            errors[init].append(algo.gradient_descent()[-1])
            sigma_min[init].append(algo.sigma_min)
            cond[init].append(algo.sigma_min/algo.sigma_max)

            if optim == GD_ON_U:
                vector_values = np.concatenate([vector_values, np.concatenate(network.clients[0].U)])
            elif optim == GD_ON_V:
                vector_values = np.concatenate([vector_values, np.concatenate(network.clients[0].V)])


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "-", "BI_SMART": "--", "ORTHO": ":"}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for init in inits:
        x, y = zip(*sorted(zip(sigma_min[init], np.log10(errors[init]))))
        axs.plot(np.array(x) ** 2, y, color=init_colors[init], linestyle=init_linestyle[init])

    init_legend = [Line2D([0], [0], linestyle="-", color=init_colors["SMART"], lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color=init_colors["BI_SMART"], lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color=init_colors["ORTHO"], lw=2, label='ortho')]

    l2 = axs.legend(handles=init_legend, loc='center right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0})$", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)
    title = f"../pictures/convergence_vs_sigma_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    x = sorted(cond["SMART"])
    error_at_optimal_solution = algo.compute_exact_solution()
    error_optimal = np.mean([np.linalg.norm(client.S - client.S_star, ord='fro') ** 2 / 2 for client in network.clients])
    y = [np.log10(error_at_optimal_solution) for i in x]
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in x]
    if USE_MOMENTUM:
        axs.plot(np.array(x) ** 1, y, color=COLORS[3], lw=3)
        if error_optimal != 0:
            axs.plot(np.array(x) ** 1, z, color=COLORS[2], lw=3)
    else:
        axs.plot(np.array(x) ** 2, y, color=COLORS[3], lw=3)
        if error_optimal != 0:
            axs.plot(np.array(x) ** 2, z, color=COLORS[2], lw=3)
    for init in inits:
        x, y = zip(*sorted(zip(cond[init], np.log10(errors[init]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=init_colors[init], linestyle=init_linestyle[init])
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=init_colors[init], linestyle=init_linestyle[init])
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)

    init_legend = init_legend = [Line2D([0], [0], linestyle="-", color=init_colors["SMART"], lw=2, label='smart init'),
                   Line2D([0], [0], linestyle="--", color=init_colors["BI_SMART"], lw=2, label='bismart init'),
                   Line2D([0], [0], linestyle=":", color=init_colors["ORTHO"], lw=2, label='ortho'),
                   Line2D([0], [0], linestyle="-", color=COLORS[3], lw=3, label='Error at opt. solution')]
    if error_optimal != 0:
        init_legend.append(Line2D([0], [0], linestyle="-", color=COLORS[2], lw=3, label='Error at true solution'))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_vs_cond_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


    plt.figure(figsize=(6, 4))
    plt.hist(np.log(np.abs(vector_values)), bins=15, alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    title = f"../pictures/hist_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')

