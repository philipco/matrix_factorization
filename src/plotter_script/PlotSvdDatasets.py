"""
Created by Constantin Philippenko, 11th December 2023.

X-axis: dimension index. Y-axis:  value of the corresponding eigenvalues..
Goal: plot the SVD of each dataset.
"""
import numpy as np
from matplotlib import pyplot as plt

from src.Network import Network
from src.utilities.MatrixUtilities import compute_svd
from src.utilities.data.DatasetsSettings import *

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

FONTSIZE=27

def plot_svd(svd, all_datasets):
    fig, axs = plt.subplots(1, 1, figsize=(9, 3))
    for dataset in all_datasets:
        axs.plot(np.log10(svd[dataset]), lw=3, label=dataset)
    axs.set_ylabel(r"$\log_{10}(\sigma_k)$", fontsize=FONTSIZE)
    axs.set_xlabel(r"$k$", fontsize=FONTSIZE)
    plt.legend(loc="lower right", fontsize=FONTSIZE, borderaxespad=0.1, labelspacing=0,
                        handletextpad=0.2)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.savefig("../../pictures/all_svd.pdf", dpi=600, bbox_inches='tight')

if __name__ == '__main__':

    svd = {}
    all_datasets = ["synth", "mnist", "celeba", "w8a"]
    for dataset_name in all_datasets:
        print(f"= {dataset_name} =")
        noise = NOISE[dataset_name] if dataset_name == "synth" else 0
        network = Network(NB_CLIENTS[dataset_name], 200, 200, RANK_S[dataset_name],
                          LATENT_DIMENSION[dataset_name], noise=NOISE[dataset_name], dataset_name=dataset_name)
        S = np.concatenate([client.S for client in network.clients])
        print(f"{dataset_name}'s shape: {S.shape}")
        svd[dataset_name] = compute_svd(S)
    plot_svd(svd, all_datasets)