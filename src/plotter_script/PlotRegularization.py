"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds

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

NB_EPOCHS = 500
NB_CLIENTS = 1

NB_RUNS = 30

USE_MOMENTUM = False
L1_COEF = 0
NOISE = 10**-2

FONTSIZE=9

if __name__ == '__main__':

    optim = GD_ON_U
    regularization = [0, 10**-5, 10**-3, 10**-1]

    errors = {}
    sigma_min = []
    inits = ["SMART"]
    cond = {}
    for r in regularization:
        errors[r] = []
        cond[r] = []

    for init in inits:
        print(f"=== {init} ===")
        for r in regularization:
            print(f"=== C = {r}")
            network = Network(NB_CLIENTS, 100, 100, 5, plunging_dimension = 5, noise=NOISE)
            sigma = []
            error = []
            for k in range(NB_RUNS):
                algo = optim(network, NB_EPOCHS, 0.01, init, use_momentum=USE_MOMENTUM, l1_coef=L1_COEF, l2_coef=r)
                # algo.compute_exact_solution(L1_COEF, r)

                # algo.__descent_initialization__()
                #
                # algo.__compute_step_size__()
                errors[r].append(algo.run()[-1])
                sigma.append(algo.sigma_min)
                cond[r].append(algo.sigma_min/algo.sigma_max)
                if cond[r][-1]**-1 < 5:
                    print(algo.__F__())
                    svd = svds(network.clients[0].U @ network.clients[0].V.T, k=5, which='LM')[1]
                    svd_true = [100 for i in range(5)]
                    print(svds(network.clients[0].S_star - network.clients[0].U @ network.clients[0].V.T, k=5, which='LM')[1])
                    print(svds(network.clients[0].S - network.clients[0].U @ network.clients[0].V.T, k=5, which='LM')[1])
                    # print(np.sum((svd_true - svd)**2))


    COLORS = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    # x = sorted(cond[overparametrization[0]])
    # error_at_optimal_solution = algo.compute_exact_solution()
    # error_optimal = np.mean(
    #     [np.linalg.norm(client.S - client.S_star, ord='fro') ** 2 / 2 for client in network.clients])
    # y = [np.log10(error_at_optimal_solution) for i in x]
    # if error_optimal != 0:
    #     z = [np.log10(error_optimal) for i in x]
    # if USE_MOMENTUM:
    #     axs.plot(np.array(x) ** 1, y, color=COLORS[0], lw=3)
    #     if error_optimal != 0:
    #         axs.plot(np.array(x) ** 1, z, color=COLORS[1], lw=3)
    # else:
    #     axs.plot(np.array(x) ** 2, y, color=COLORS[0], lw=3)
    #     if error_optimal != 0:
    #         axs.plot(np.array(x) ** 2, z, color=COLORS[1], lw=3)
    for k in range(len(regularization)):
        x, y = zip(*sorted(zip(cond[regularization[k]], np.log10(errors[regularization[k]]))))
        if USE_MOMENTUM:
            axs.plot(np.array(x) ** 1, y, color=COLORS[k], linestyle="-", label=r"reg $= {:.0e}$".format(regularization[k]))
            axs.set_xlabel(r"$\sigma_{\mathrm{min}}(\mathbf{V_0}) / \sigma_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)
        else:
            axs.plot(np.array(x) ** 2, y, color=COLORS[k], linestyle="-", label=r"reg $= {:.0e}$".format(regularization[k]))
            axs.set_xlabel(r"$\sigma^2_{\mathrm{min}}(\mathbf{V_0}) / \sigma^2_{\mathrm{max}}(\mathbf{V_0})$",
                           fontsize=FONTSIZE)


    axs.legend(fontsize=FONTSIZE)

    axs.set_ylabel("Log(Relative error)", fontsize=FONTSIZE)
    title = f"../pictures/convergence_regularization_N{network.nb_clients}_r{network.plunging_dimension}_{algo.variable_optimization()}"
    if USE_MOMENTUM:
        title += f"_momentum"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


