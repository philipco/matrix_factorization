"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.Network import Network
from src.algo.GradientDescent import GD_ON_U
from src.utilities.data.DatasetsSettings import *

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

DATASET_NAME = "synth"
L1_COEF = 0 #10**-2
L2_COEF = 0 #10**-3
NUC_COEF = 0
# USE_MOMENTUM = True

FONTSIZE=9

if __name__ == '__main__':

    print(f"= {DATASET_NAME} =")
    network = Network(NB_CLIENTS[DATASET_NAME], 60, 60, RANK_S[DATASET_NAME],
                      60, LATENT_DIMENSION[DATASET_NAME], noise=NOISE[DATASET_NAME],
                      dataset_name=DATASET_NAME, m=20)

    optimization = GD_ON_U
    errors = {}
    error_at_optimal_solution = {}
    error_back_to_complete_space = {}

    labels = {"SMART": r"$\alpha=0$", "POWER": r"$\alpha=1$"}
    inits = ["SMART"] #, "POWER"]

    # RANDOM initialization for optimization on U,V
    # algo = GD_ON_U(network, NB_EPOCHS, 0.01, "POWER")
    # plt.show()
    # algo.elastic_net()
    # plt.imshow((network.clients[0].U @ network.clients[0].V.T))
    # plt.title("Elastic net")
    # plt.show()
    # plt.imshow(network.clients[0].U)
    # plt.title("Elastic net: U")
    # plt.show()
    # algo.compute_exact_solution(L1_COEF, L2_COEF, NUC_COEF)
    # if dataset_name == 'celeba':
    #     plt.imshow(np.transpose(network.clients[0].S.reshape(-1,3,32), (0, 2, 1))[:64])
    #     plt.title("Two first images of client 0")
    #     plt.show()
    #     im1 = (network.clients[0].U @ network.clients[0].V.T).reshape(-1,3,32)
    #     plt.imshow(np.transpose(im1, (0, 2, 1))[:64])
    #     plt.title("Exact solution for the two first images of client 0", fontsize=FONTSIZE)
    # plt.imshow(network.clients[0].U @ network.clients[0].V.T)
    # plt.show()
    # plt.imshow(network.clients[0].V)
    # plt.title("V", fontsize=FONTSIZE)
    # plt.show()
    # plt.imshow(network.clients[0].U)
    # plt.title("U", fontsize=FONTSIZE)
    # plt.show()
    # algo.run()
    # plt.imshow(network.clients[0].U @ network.clients[0].V.T)
    # plt.title("Gradient descent", fontsize=FONTSIZE)
    # plt.show()

    for init in inits:
        print(f"\t== {init} ==")
        for use_momentum in [False, True]:
            print(f"\t===> Use momentum: {use_momentum}")
            algo = optimization(network, NB_EPOCHS[DATASET_NAME], 0.01, init, L1_COEF, L2_COEF, NUC_COEF,
                                use_momentum=use_momentum)
            key = init + "_momentum" if use_momentum else init
            errors[key] = algo.run()
            error_at_optimal_solution[init] = algo.compute_exact_solution(L1_COEF, L2_COEF, NUC_COEF)
            error_back_to_complete_space[init] = algo.back_to_full_space(L1_COEF, L2_COEF, NUC_COEF)
            print(f"{init}\terror min:", errors[init][-1])

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    init_colors = {"SMART": COLORS[0], "BI_SMART": COLORS[1], "ORTHO": COLORS[4], "POWER": COLORS[5]}

    fig, axs = plt.subplots(1, 1, figsize=(3, 4))

    for init in inits:
        axs.plot(np.log10(errors[init]), color=init_colors[init], linestyle="--")
        axs.plot(np.log10(errors[init + "_momentum"]), color=init_colors[init], linestyle="-.")
        x = np.linspace(0, len(errors[init]), num=10)
        z = [np.log10(error_at_optimal_solution[init]) for i in x]
        axs.plot(x, z, color=init_colors[init], marker="*")

    ## Optimal error. ###
    S_stacked = np.concatenate([client.uncompressed_S for client in network.clients])
    _, singular_values, _ = scipy.linalg.svd(S_stacked)

    error_optimal = 0.5 * np.sum([singular_values[i] ** 2 for i in range(network.plunging_dimension + 1,
                                                                         min(np.sum(
                                                                             [c.nb_samples for c in network.clients]),
                                                                             network.sub_dimension))])
    if error_optimal != 0:
        z = [np.log10(error_optimal) for i in errors["SMART"]]
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

    if DATASET_NAME == "synth":
        l2 = axs.legend(handles=init_legend, loc='upper right', fontsize=FONTSIZE)
        axs.add_artist(l2)
    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    title = f"../pictures/{DATASET_NAME}_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if NOISE[DATASET_NAME] != 0:
        title += f"_eps{NOISE[DATASET_NAME]}"
    if algo.l1_coef != 0:
        title += f"_lasso{L1_COEF}"
    if algo.l2_coef != 0:
        title += f"_ridge{L2_COEF}"
    if algo.nuc_coef != 0:
        title += f"nuc{NUC_COEF}"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')
