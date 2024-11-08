"""
Created by Constantin Philippenko, 11th December 2023.

X-axis: iteration index. Y-axis:  logarithm of the loss F after 1000 local iterations.
Goal: illustrate on real-life datasets how the algorithm behaves in practice.
"""
import argparse

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U
from src.utilities.MatrixUtilities import compute_optimal_error
from src.utilities.data.DatasetsSettings import *

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

L1_COEF = 0 #10**-2
L2_COEF = 0 #10**-3
NUC_COEF = 0

FONTSIZE=9

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        required=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    print(f"= {dataset_name} =")
    network = Network(NB_CLIENTS[dataset_name], 200, 200, RANK_S[dataset_name],
                      LATENT_DIMENSION[dataset_name], noise=NOISE[dataset_name], dataset_name=dataset_name, m=20)

    optimization = GD_ON_U
    errors = {}
    error_at_optimal_solution = {}

    labels = {"power0": r"$\alpha=0$", "power1": r"$\alpha=1$"}
    inits = ["power0", "power1"]

    for init in inits:
        print(f"\t== {init} ==")
        for use_momentum in [False, True]:
            print(f"\t===> Use momentum: {use_momentum}")
            algo = optimization(network, NB_EPOCHS[dataset_name], init, L1_COEF, L2_COEF, NUC_COEF,
                                use_momentum=use_momentum)
            key = init + "_momentum" if use_momentum else init
            errors[key] = algo.run()
            error_at_optimal_solution[init] = algo.compute_exact_solution(L1_COEF, L2_COEF, NUC_COEF)
            print(f"{init}\terror min:", errors[init][-1])

    COLORS = ["tab:blue", "tab:brown", "tab:green"]
    init_colors = {"power0": COLORS[0], "power1": COLORS[1]}

    fig, axs = plt.subplots(1, 1, figsize=(3, 4))

    for init in inits:
        axs.plot(np.log10(errors[init]), color=init_colors[init], linestyle="--")
        axs.plot(np.log10(errors[init + "_momentum"]), color=init_colors[init], linestyle="-.")
        x = np.linspace(0, len(errors[init]), num=10)
        z = [np.log10(error_at_optimal_solution[init]) for i in x]
        axs.plot(x, z, color=init_colors[init], marker="*")

    ## Optimal error. ###
    error_optimal = compute_optimal_error([client.S for client in network.clients],
                                          [c.nb_samples for c in network.clients], network.dim,
                                          network.plunging_dimension)

    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in errors["power0"]]
        axs.plot(z, color=COLORS[2], lw=3)

    init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                          lw=3, label=labels[init]) for init in inits]
    if error_optimal != 0:
        init_legend.append(
            Line2D([0], [0], linestyle="-", color=COLORS[2], lw=2, label=r'$ \sum_{i>r} \sigma_i^2 / 2$'))
    init_legend.append(Line2D([0], [0], linestyle="-", color='black', lw=2, marker="*",
                              label="Exact solution"))
    init_legend.append(Line2D([0], [0], linestyle="--", color='black', lw=2, label="Gradient descent (GD)"))
    init_legend.append(Line2D([0], [0], linestyle="-.", color='black', lw=2, label="GD w. momentum"))

    if dataset_name == "synth":
        l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
        axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    title = f"../../pictures/{dataset_name}_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if NOISE[dataset_name] != 0:
        title += f"_eps{NOISE[dataset_name]}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if algo.nuc_coef != 0:
        title += f"nuc{NUC_COEF}"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')
