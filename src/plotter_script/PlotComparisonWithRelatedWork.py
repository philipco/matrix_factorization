"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
import scipy
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

NB_EPOCHS = 300
NB_CLIENTS = 10

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0

NB_RUNS = 20
NOISE = 10**-6

FONTSIZE = 9

if __name__ == '__main__':

    # network = Network(NB_CLIENTS, None, None, None, 100, noise=0,
    #                   image_name="cameran")
    network = Network(NB_CLIENTS, 100, 100, 5, 10, noise=NOISE)

    inits = ["SMART", "POWER"]
    labels = {"SMART": r"$\alpha=0$", "POWER": r"$\alpha=1$",  "LocalPower": 'LocalPower'}
    algo_name = ["LocalPower"]

    errors = {name: [] for name in inits + algo_name}
    error_at_optimal_solution = {name: [] for name in inits}
    cond = {name: [] for name in inits + algo_name}
    related_work = [DistributedPowerMethod]

    # Running related work no using GD.
    for optim in related_work:
        for k in range(NB_RUNS):
            algo = optim(network, 1, 0.01, "RANDOM", 1)
            algo_errors = algo.run()
            errors[algo.name()].append(algo_errors[-1])
            cond[algo.name()].append(algo.sigma_min / algo.sigma_max)

    # Running algorithms based on GD.
    optim = GD_ON_U
    for init in inits:
        print(f"=== {init} ===")
        for k in range(NB_RUNS):
            algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=False, l1_coef=L1_COEF, l2_coef=L2_COEF)
            errors[init].append(algo.run()[-1])
            cond[init].append(algo.sigma_min / algo.sigma_max)
            error_at_optimal_solution[init].append(algo.compute_exact_solution(L1_COEF, L2_COEF))

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "--", "BI_SMART": "--", "ORTHO": ":", "POWER": "--"}  # (0, (3, 1, 1, 1))}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5],
                   "LocalPower": COLORS[6]}


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
    x, y = zip(*sorted(zip(cond["SMART"], np.log10(errors["LocalPower"]))))
    axs.plot(np.array(x) ** 2, y, color=init_colors["LocalPower"], linestyle="-", label="LocalPower", marker="^")

    ## Optimal error. ###
    S_stacked = np.concatenate([client.S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                         min(NB_CLIENTS * network.nb_samples,
                                                                             network.dim))])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in cond["SMART"]]
        if USE_MOMENTUM:
            axs.plot(np.array(cond["SMART"]) ** 1, z, color=COLORS[2], lw=2)
        else:
            axs.plot(np.array(cond["SMART"]) ** 2, z, color=COLORS[2], lw=2)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=2, label=labels[init]) for init in inits]
    for init in algo_name:
        init_legend.append(Line2D([0], [0], color=init_colors[init], linestyle="-",
                                  lw=2, label=labels[init], marker="^"))
    if error_optimal != 0:
        init_legend.append(
            Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, label="Exact solution"))
    init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=2, label="Gradient descent"))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/related_work_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if NOISE != 0:
        title += f"_eps{NOISE}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


