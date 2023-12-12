"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt

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

NB_EPOCHS = 5000
NB_CLIENTS = 1

FONTSIZE=9

if __name__ == '__main__':

    network = Network(NB_CLIENTS, 100, 100, 5, 6)
    algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "SMART_FOR_GD_ON_U")
    errorU = algoU.gradient_descent()
    random_algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "RANDOM")
    random_errorU = random_algoU.gradient_descent()
    bismart_algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "BI_SMART_FOR_GD_ON_U")
    bismart_errorU = bismart_algoU.gradient_descent()
    ortho_algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "ORTHO_FOR_GD_ON_U")
    ortho_errorU = ortho_algoU.gradient_descent()
    algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "SMART")
    errorV = algoV.gradient_descent()
    random_algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "RANDOM")
    random_errorV = random_algoV.gradient_descent()
    bismart_algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "BI_SMART")
    bismart_errorV = bismart_algoV.gradient_descent()
    ortho_algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "ORTHO")
    ortho_errorV = ortho_algoV.gradient_descent()
    randomAlternateGD = AlternateGD(network, NB_EPOCHS, 0.01, "RANDOM")
    error_random_alternate = randomAlternateGD.gradient_descent()
    alternateGD = AlternateGD(network, NB_EPOCHS, 0.01, "SMART")
    error_alternate = alternateGD.gradient_descent()
    print(f"\n{algoU.name()}:")
    print("First value of F:", errorU[0])
    print("Last value of F:", errorU[-1])

    print(f"\n{algoV.name()}:")
    print("First value of F:", errorV[0])
    print("Last value of F:", errorV[-1])

    print(f"\n{randomAlternateGD.name()}:")
    print("First value of F:", error_random_alternate[0])
    print("Last value of F:", error_random_alternate[-1])

    print(f"\n{alternateGD.name()}:")
    print("First value of F:", error_alternate[0])
    print("Last value of F:", error_alternate[-1])

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]


    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(np.log10(error_random_alternate), label="gd on U, V, random init.", color=COLORS[0], linestyle="-.")
    axs.plot(np.log10(error_alternate), label="gd on U, V, smart init.", color=COLORS[0])
    axs.plot(np.log10(random_errorV), label="gd on V, random init.", color=COLORS[1], linestyle="-.")
    axs.plot(np.log10(errorV), label="gd on V, smart init.".format(algoV.smallest_sv[1]), color=COLORS[1]) #, sv(V)={:.2f}
    axs.plot(np.log10(bismart_errorV), label="gd on V, bismart init.", color=COLORS[1], linestyle="--")
    axs.plot(np.log10(ortho_errorV), label=f"gd on V, ortho init.", color=COLORS[1], linestyle=":")
    axs.plot(np.log10(random_errorU), label="gd on U, random init.", color=COLORS[2], linestyle="-.")
    axs.plot(np.log10(errorU), label="gd on U, smart init.".format(algoU.smallest_sv[0]), color=COLORS[2]) #, sv(U)={:.2f}
    axs.plot(np.log10(bismart_errorU), label="gd on U, bismart init.", color=COLORS[2], linestyle="--")
    axs.plot(np.log10(ortho_errorU), label=f"gd on U, ortho init.", color=COLORS[2], linestyle=":")
    # axs.plot([np.log10(np.exp(- 2 * i * step_size * svd(result_params[0][0])[1][-1])) for i in range(len(error_S_star))], label="th. ")
    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)
    axs.legend(fontsize=FONTSIZE, loc='center right')
    plt.savefig(f"5-convergence_N{network.nb_clients}_r{network.plunging_dimension}.pdf", dpi=600, bbox_inches='tight')

