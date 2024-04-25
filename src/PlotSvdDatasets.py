"""
Created by Constantin Philippenko, 11th December 2023.
"""
import numpy as np
from matplotlib import pyplot as plt

from src.Client import Network
from src.MatrixUtilities import compute_svd
from src.utilities.data.DatasetsSettings import *

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})



FONTSIZE=9


def plot_svd(svd, all_datasets):
    fig, axs = plt.subplots(1, 1, figsize=(3, 4))
    for dataset in all_datasets:
        axs.plot(np.log(svd[dataset]), lw=3, label=dataset)
    axs.set_ylabel(r"$\log(\sigma_k)$", fontsize=15)
    axs.set_xlabel(r"$k$", fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    title = f"../pictures/all_svd"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')

if __name__ == '__main__':

    svd = {}
    all_datasets = ["synth", "mnist", "celeba", "w8a"]
    for dataset_name in all_datasets:
        noise = 10 ** -6 if dataset_name == "synth" else 0
        network = Network(NB_CLIENTS[dataset_name], 100, 100, RANK_S[dataset_name],
                          LATENT_DIMENSION[dataset_name], noise=NOISE[dataset_name], dataset_name=dataset_name)
        S = np.concatenate([client.S for client in network.clients])
        svd[dataset_name] = compute_svd(S)
    plot_svd(svd, all_datasets)
