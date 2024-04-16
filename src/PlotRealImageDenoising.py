"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Client import Network
from src.algo.GradientDescent import GD_ON_U, GD_ON_V

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 200
NB_CLIENTS = 10
L1_COEF = 0
L2_COEF = 0

FONTSIZE=9

if __name__ == '__main__':

    dataset_name = "celeba"
    network = Network(NB_CLIENTS, None, None, None, 40, noise=0,
                      image_name=dataset_name)

    optimization = GD_ON_U
    errors = {}
    error_at_optimal_solution = {}

    labels = {"SMART": r"$\alpha=1$", "POWER": r"$\alpha=3$"}
    inits = ["SMART", "POWER"]

    # RANDOM initialization for optimization on U,V
    algo = GD_ON_U(network, NB_EPOCHS, 0.01, "POWER")
    algo.compute_exact_solution(L1_COEF, L2_COEF)
    plt.imshow(network.clients[0].S)
    plt.show()
    plt.imshow(network.clients[0].U @ network.clients[0].V.T)
    plt.title("Exact solution", fontsize=FONTSIZE)
    plt.show()
    algo.run()
    plt.imshow(network.clients[0].U @ network.clients[0].V.T)
    plt.title("Gradient descent", fontsize=FONTSIZE)
    plt.show()

    for init in inits:
        print(f"\t== {init} ==")
        algo = optimization(network, NB_EPOCHS, 0.01, init)
        errors[init] = algo.run()
        error_at_optimal_solution[init] = algo.compute_exact_solution(0, 0)
        print(f"{init}\terror min:", errors[init][-1])

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    optim_colors = {"UV": COLORS[0], "V": COLORS[1], "U": COLORS[2]}
    init_linestyle = {"RANDOM": "-.", "SMART": "-", "BI_SMART": "--", "ORTHO": ":"}

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]

    init_linestyle = {"SMART": "--", "BI_SMART": "--", "ORTHO": ":", "POWER": "--"}  # (0, (3, 1, 1, 1))}
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    for init in inits:
        axs.plot(np.log10(errors[init]), color=init_colors[init], linestyle=init_linestyle[init])
        z = [np.log10(error_at_optimal_solution[init]) for i in errors[init]]
        axs.plot(z, color=init_colors[init])

    ## Optimal error. ###
    S_stacked = np.concatenate([client.S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                         min(np.sum(
                                                                             [c.nb_samples for c in network.clients]),
                                                                             network.dim))])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in errors["SMART"]]
        axs.plot(z, color=COLORS[2], lw=2)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=2, label=labels[init]) for init in inits]
    if error_optimal != 0:
        init_legend.append(
            Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, label="Exact solution"))
    init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=2, label="Gradient descent"))

    l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
    axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/{dataset_name}_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"

    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')
