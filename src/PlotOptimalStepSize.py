"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt

from src.Client import Network
from src.algo.GradientDescent import GD_ON_V

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

FONTSIZE=9

if __name__ == '__main__':


    network = Network(NB_CLIENTS, 100, 100, 5, 5)

    # optimizations = {"UV": AlternateGD, "V": GD_ON_V, "U": GD_ON_U}
    optim = GD_ON_V
    step_size_factors = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    errors = {}
    sigma_min = []
    inits = ["SMART"]
    cond = {}
    for C in step_size_factors:
        errors[C] = []
        cond[C] = []

    for init in inits:
        print(f"=== {init} ===")
        for C in step_size_factors:
            print(f"=== C = {C}")
            sigma = []
            error = []
            for k in range(NB_RUNS):
                algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
                algo.step_size *= C
                errors[C].append(algo.run()[-1])
                sigma.append(algo.sigma_min)
                cond[C].append(algo.sigma_min/algo.sigma_max)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for k in range(len(step_size_factors)):
        x, y = zip(*sorted(zip(cond[step_size_factors[k]], np.log10(errors[step_size_factors[k]]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=COLORS[k], linestyle="-", label=r"$\gamma \leftarrow \times {0}$".format(step_size_factors[k]))
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=COLORS[k], linestyle="-", label=r"$\gamma \leftarrow \times {0}$".format(step_size_factors[k]))
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)


    axs.legend(fontsize=FONTSIZE)

    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_vs_stepsize_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


