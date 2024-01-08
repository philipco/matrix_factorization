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
NB_CLIENTS = 5

NB_RUNS = 50

USE_MOMENTUM = False
L1_COEF = 0
L2_COEF = 0

FONTSIZE=9

if __name__ == '__main__':


    network = Network(NB_CLIENTS, 100, 100, 5, 5)

    # optimizations = {"UV": AlternateGD, "V": GD_ON_V, "U": GD_ON_U}
    optim = GD_ON_V
    step_size_factors = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]

    errors = []
    sigma_min = []
    inits = ["SMART"]

    for init in inits:
        print(f"=== {init} ===")
        for C in step_size_factors:
            print(f"=== C = {C}")
            sigma = []
            error = []
            for k in range(NB_RUNS):
                algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=L2_COEF)
                algo.step_size *= C
                error.append(algo.gradient_descent()[-1])
                sigma.append(algo.sigma_min)
            sigma_min.append(sigma)
            errors.append(error)


    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for k in range(len(step_size_factors)):
        x, y = zip(*sorted(zip(sigma_min[k], np.log10(errors[k]))))
        axs.plot(np.array(x) ** 2, y, color=COLORS[k], linestyle="-", label=r"$C = {0}$".format(step_size_factors[k]))


    axs.legend(fontsize=FONTSIZE)
    axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{D_*} \mathbf{U_*}^\top \mathbf{\Phi})$", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)
    plt.savefig(f"../pictures/convergence_vs_stepsize_N{network.nb_clients}_r{network.plunging_dimension}.pdf", dpi=600, bbox_inches='tight')


