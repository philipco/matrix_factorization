"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 1000
NB_CLIENTS = 1

NB_RUNS = 30

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0
NOISE = 0

FONTSIZE=9

if __name__ == '__main__':

    # optimizations = {"UV": AlternateGD, "V": GD_ON_V, "U": GD_ON_U}
    optim = GD_ON_U
    overparametrization = [0,1,2,4,8]

    errors = {}
    sigma_min = []
    inits = ["SMART"]
    cond = {}
    for r in overparametrization:
        errors[r] = []
        cond[r] = []

    for init in inits:
        print(f"=== {init} ===")
        for r in overparametrization:
            print(f"=== C = {r}")
            network = Network(NB_CLIENTS, 100, 100, 5, plunging_dimension = 5 + r, noise=NOISE)
            sigma = []
            error = []
            for k in range(NB_RUNS):
                algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
                errors[r].append(algo.run()[-1])
                sigma.append(algo.sigma_min)
                cond[r].append(algo.sigma_min/algo.sigma_max)


    COLORS = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    x = sorted(cond[overparametrization[0]])
    error_at_optimal_solution = algo.compute_exact_solution(L1_COEF, L2_COEF)
    error_optimal = np.mean(
        [np.linalg.norm(client.S - client.S_star, ord='fro') ** 2 / 2 for client in network.clients])
    y = [np.log10(error_at_optimal_solution) for i in x]
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in x]
    if USE_MOMENTUM:
        axs.plot(np.array(x) ** 1, y, color=COLORS[0], lw=3)
        if error_optimal != 0:
            axs.plot(np.array(x) ** 1, z, color=COLORS[1], lw=3)
    else:
        axs.plot(np.array(x) ** 2, y, color=COLORS[0], lw=3)
        if error_optimal != 0:
            axs.plot(np.array(x) ** 2, z, color=COLORS[1], lw=3)
    for k in range(len(overparametrization)):
        x, y = zip(*sorted(zip(cond[overparametrization[k]], np.log10(errors[overparametrization[k]]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=COLORS[k], linestyle="-", label=r"$r = r_* +{0}$".format(overparametrization[k]))
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=COLORS[k], linestyle="-", label=r"$r = r_* +{0}$".format(overparametrization[k]))
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)


    axs.legend(fontsize=FONTSIZE)

    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_overparametrized_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


