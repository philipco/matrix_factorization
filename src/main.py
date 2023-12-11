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

NB_EPOCHS = 10000

FONTSIZE=14

if __name__ == '__main__':

    network = Network(1, 100, 100, 5, 6)
    algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "SMART_FOR_GD_ON_U")
    errorU = algoU.gradient_descent()
    random_algoU = GD_ON_U(network, NB_EPOCHS, 0.01, "RANDOM")
    random_errorU = random_algoU.gradient_descent()
    algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "SMART")
    errorV = algoV.gradient_descent()
    random_algoV = GD_ON_V(network, NB_EPOCHS, 0.01, "RANDOM")
    random_errorV = random_algoV.gradient_descent()
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

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(np.log10(error_alternate), label="w. smart init.")
    axs.plot(np.log10(error_random_alternate), label="w.o. smart init.")
    axs.plot(np.log10(errorV), label="gd on V, w. smart init.")
    axs.plot(np.log10(random_errorV), label="gd on V, w.o. smart init.")
    axs.plot(np.log10(errorU), label="gd on U, w. smart init.")
    axs.plot(np.log10(random_errorU), label="gd on U, w.o. smart init.")
    # axs.plot([np.log10(np.exp(- 2 * i * step_size * svd(result_params[0][0])[1][-1])) for i in range(len(error_S_star))], label="th. ")
    axs.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    # axs.set_ylabel("$\\frac{1}{2N} \sum_{i=1}^N \|\mathbf{S} - \mathbf{U} \mathbf{V}^\\top \|^2_ F$", fontsize=FONTSIZE)
    axs.set_ylabel("Relative error", fontsize=FONTSIZE)
    # axs.set_title("Error to $\mathbf{S}$", fontsize=FONTSIZE)
    axs.legend(fontsize=FONTSIZE)
    plt.savefig(f"2_convergence_N{network.nb_clients}.pdf", dpi=600, bbox_inches='tight')

