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

NB_EPOCHS = 250
NB_CLIENTS = 1

NB_RUNS = 15

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0

FONTSIZE=9

if __name__ == '__main__':

    optim = GD_ON_U
    missing_value_ratio = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

    errors = {}
    sigma_min = []
    inits = ["SMART"]
    cond = {}
    for ratio in missing_value_ratio:
        errors[ratio] = []
        cond[ratio] = []

    for init in inits:
        print(f"=== {init} ===")
        for ratio in missing_value_ratio:
            network = Network(NB_CLIENTS, 100, 100, 5, 5, missing_value=ratio)
            print(f"=== ratio = {ratio}")
            sigma = []
            error = []
            for k in range(NB_RUNS):
                algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
                errors[ratio].append(algo.run()[-1])
                sigma.append(algo.sigma_min)
                cond[ratio].append(algo.sigma_min / algo.sigma_max)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for k in range(len(missing_value_ratio)):
        x, y = zip(*sorted(zip(cond[missing_value_ratio[k]], np.log10(errors[missing_value_ratio[k]]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=COLORS[k], linestyle="-", label=r"ratio m.v. $=  {0}$".format(missing_value_ratio[k]))
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=COLORS[k], linestyle="-", label=r"ratio m.v. $= {0}$".format(missing_value_ratio[k]))
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)


    axs.legend(fontsize=FONTSIZE)

    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_vs_ratio_mv_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


