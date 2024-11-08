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
from src.algo.GradientDescent import GD_ON_U, AlternateGD, GD
from src.algo.PowerMethods import LocalPower
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


def print_table_communication(datasets, nb_communication, keys):
    columns = "l"*(len(datasets)+1)
    table = ("\\begin{table}[]\n"
             "\\begin{tabular}{" + "{0}\n".format(columns)) + "}\n"
    head = r"$\# communication$" + "".join([f'& {name}' for name in datasets]) #
    head += "\\\\ \n"
    table += head
    for key in keys:
        rows = f"{key}"
        for name in datasets:
            nb = nb_communication[name][key]
            if nb == 100:
                rows += '& $+100$'
            else:
                rows += '& {0}'.format(nb)
        rows += "\\\\ \n"
        table += rows

    table += ("\end{tabular}\n"
              "\end{table}")
    print(table)

def print_table(datasets, errors, error_optimal, keys):


    columns = "l"*(len(datasets)+1)
    table = ("\\begin{table}[]\n"
             "\\begin{tabular}{" + "{0}\n".format(columns)) + "}\n"
    head = r"$log_{10}(\epsilon - \epsilon_{\mathrm{min}})$" + "".join([f'& {name}' for name in datasets]) #
    head += "\\\\ \n"
    table += head
    for key in keys:
        rows = f"{key}"
        for name in datasets:
            excess_loss = errors[name][key][-1] - error_optimal[name]
            if excess_loss == 0:
                rows += '& $=$'
            else:
                rows += '& {:.2f}'.format(np.log10(excess_loss))
        rows += "\\\\ \n"
        table += rows

    table += ("\end{tabular}\n"
              "\end{table}")
    print(table)

if __name__ == '__main__':

    keys = ["power0 GD", "power1 GD", "GD", "Alternate GD"]
    labels = {"power0 GD": r"$\alpha=0$", "power1 GD": r"$\alpha=1$", "GD": "GD", "Alternate GD": "Alternate GD"}
    inits = ["power0", "power1"]
    related_work = {"Alternate GD": AlternateGD, "GD": GD}

    EPS = {"synth": -7, "mnist": 5.5, "celeba": 5, "w8a": 5}

    datasets = ["synth", "w8a", "mnist", "celeba"]
    errors = {name: {} for name in datasets}
    nb_commmunications = {name: {} for name in datasets}
    error_at_optimal_solution = {name: {} for name in datasets}
    error_optimal = {}
    for dataset_name in datasets:

        print(f"= {dataset_name} =")
        # NB_CLIENTS[dataset_name]
        network = Network(NB_CLIENTS[dataset_name], 200, 200, RANK_S[dataset_name],
                          LATENT_DIMENSION[dataset_name], noise=NOISE[dataset_name], dataset_name=dataset_name, m=20)

        ## Optimal error. ###
        error_optimal[dataset_name] = compute_optimal_error([client.S for client in network.clients],
                                                            [c.nb_samples for c in network.clients], network.dim,
                                                            network.plunging_dimension)

        for label, optim in related_work.items():
            print(f"\t== {label} ==")
            algo = optim(network, NB_EPOCHS[dataset_name], L1_COEF, L2_COEF, NUC_COEF)
            errors[dataset_name][label], nb_commmunications[dataset_name][label] = algo.run(eps=EPS[dataset_name], optimal_error=error_optimal[dataset_name])
            print(f"{label}\terror min:", errors[dataset_name][label][-1])

        for init in inits:
            print(f"\t== {init} ==")
            algo = GD_ON_U(network, NB_EPOCHS[dataset_name], init, L1_COEF, L2_COEF, NUC_COEF,
                                use_momentum=True)
            errors[dataset_name][init + " GD"], nb_commmunications[dataset_name][init + " GD"] = algo.run(eps=EPS[dataset_name], optimal_error=error_optimal[dataset_name])
            error_at_optimal_solution[dataset_name][init] = algo.compute_exact_solution(L1_COEF, L2_COEF, NUC_COEF)
            local_power = LocalPower(network, 1 if init == "power0" else 2)
            errors[dataset_name][init + " local"] = local_power.run()
            print(f"{init}\terror min:", errors[dataset_name][init + " GD"][-1])
            print(f"{init}\terror min at optimal solution:", error_at_optimal_solution[dataset_name][init])
            print(f"{init}_local\terror min:", errors[dataset_name][init + " local"][-1])

        COLORS = ["tab:blue", "tab:brown", "tab:green", "tab:orange", "tab:red"]
        init_colors = {"power0 GD": COLORS[0], "power1 GD": COLORS[1], "GD": COLORS[3], "Alternate GD": COLORS[4]}

        fig, axs = plt.subplots(1, 1, figsize=(3, 4))

        for init in inits:
            axs.plot(np.log10(errors[dataset_name][init + " GD"]), color=init_colors[init + " GD"], linestyle="-.")
            x = np.linspace(0, len(errors[dataset_name][init + " GD"]), num=10)
            z = [np.log10(error_at_optimal_solution[dataset_name][init]) for i in x]
            axs.plot(x, z, color=init_colors[init + " GD"], marker="*")

        for init in related_work.keys():
            axs.plot(np.log10(errors[dataset_name][init]), color=init_colors[init], linestyle="-.")


        if error_optimal != 0:
            z = [np.log10(error_optimal[dataset_name]) for i in errors[dataset_name]["power0 GD"]]
            axs.plot(z, color=COLORS[2], lw=3)
            print(f"\toptimal error min:", error_optimal[dataset_name])

        init_legend = [Line2D([0], [0], color=init_colors[init], linestyle="-",
                              lw=3, label=labels[init]) for init in labels.keys()]
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
        title = f"../../pictures/related_works_{dataset_name}_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_{algo.variable_optimization()}"
        plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


    print_table(datasets, errors, error_optimal, keys)
    print_table_communication(datasets, nb_commmunications, keys)