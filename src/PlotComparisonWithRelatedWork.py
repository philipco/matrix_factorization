"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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

NB_EPOCHS = 300
NB_LOCAL_EPOCHS = 10
NB_CLIENTS = 10

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0

NB_RUNS = 20
NOISE = 0*10**-2

FONTSIZE = 9

if __name__ == '__main__':

    # network = Network(NB_CLIENTS, None, None, None, 100, noise=0,
    #                   image_name="cameran")
    network = Network(NB_CLIENTS, 100, 100, 5, 6, noise=NOISE)

    inits = ["SMART", "ORTHO", "POWER"]
    labels = {"SMART": 'bismart', "ORTHO": "ortho", "POWER": 'power'}
    algo_name = ["LocalPower"]

    errors = {name: [] for name in inits + algo_name}
    error_at_optimal_solution = {name: [] for name in inits}
    cond = {name: [] for name in inits + algo_name}
    related_work = [DistributedPowerMethod]

    # Running related work no using GD.
    for optim in related_work:
        for k in range(NB_RUNS):
            algo = optim(network, NB_EPOCHS // NB_LOCAL_EPOCHS, 0.01, "RANDOM", NB_LOCAL_EPOCHS)
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

    # init_linestyle = {"SMART": "-.", "BI_SMART": "--", "ORTHO": ":"}
    algo_colors = {"SMART": COLORS[0], "ORTHO": COLORS[4], "POWER": COLORS[5], "LocalPower": COLORS[6]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    for init in inits:
        x, y = zip(*sorted(zip(cond[init], np.log10(errors[init]))))
        axs.plot(np.array(x) ** 2, y, color=algo_colors[init], linestyle="--", label=f"GD w. {labels[init]} init.")

        x, y = zip(*sorted(zip(cond[init], np.log10(error_at_optimal_solution[init]))))
        axs.plot(np.array(x) ** 2, y, color=algo_colors[init], lw=1, label=f"Exact sol. w. {labels[init]} init.")

    x, y = zip(*sorted(zip(cond["SMART"], np.log10(errors["LocalPower"]))))
    axs.plot(np.array(x) ** 2, y, color=algo_colors["LocalPower"], linestyle="-", label="LocalPower")

    axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                   fontsize=FONTSIZE)

    error_optimal = np.mean(
        [np.linalg.norm(client.S - client.S_star, ord='fro') ** 2 / 2 for client in network.clients])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in x]
        axs.plot(np.array(x) ** 2, z, color=COLORS[2], lw=2, label=r"$\|S - S_*\|_F^2 / 2$")

    axs.legend(loc='upper right', fontsize=FONTSIZE)
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


