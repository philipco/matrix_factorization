"""
Created by Constantin Philippenko, 11th December 2023.

X-axis: logarithm of the loss F after 1000 local iterations. Y-axis: cumulative distribution function.
Goal: plot the cumulative distribution function of the global objective function F, i.e. the error of reconstruction,
    for 30 different sampled Gausian matrices Phi.


"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds

from src.Network import Network

import matplotlib

from src.algo.MFInitialization import generate_gaussian_matrix

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

NB_EPOCHS = 1000
NB_CLIENTS = 1
USE_MOMENTUM = False
NB_RUNS = 30

FONTSIZE=15



def power_init_pb(network, l2_coef: int, do_orth: bool):
    S = np.concatenate([client.S for client in network.clients])
    Phi_V = [generate_gaussian_matrix(client.nb_samples, client.plunging_dimension, 1) for client in
             network.clients]
    Phi = np.concatenate(Phi_V)
    V = S.T @ S @ S.T @ Phi
    if do_orth:
        V = scipy.linalg.orth(V, rcond=0)
    X = S - S @ V @ np.linalg.pinv(V.T @ V + l2_coef * np.eye(network.plunging_dimension)) @ V.T
    return np.linalg.norm(X, ord='fro') ** 2 / 2, svds(Phi, k=1, which='LM')[1][0] / svds(Phi, k=1, which='SM')[1][0]


def smart_init_pb(network, l2_coef: int, do_orth: bool):
    S = np.concatenate([client.S for client in network.clients])
    Phi_V = [generate_gaussian_matrix(client.nb_samples, client.plunging_dimension, 1) for client in
     network.clients]
    Phi = np.concatenate(Phi_V)
    V = S.T @ Phi
    if do_orth:
        V = scipy.linalg.orth(V, rcond=0)
    X = S - S @ V @ np.linalg.pinv(V.T @ V + l2_coef * np.eye(network.plunging_dimension)) @ V.T
    return np.linalg.norm(X, ord='fro') ** 2 / 2, svds(Phi, k=1, which='LM')[1][0] / svds(Phi, k=1, which='SM')[1][0]


def compute_error(network, init, l2_coef: int, do_orth: bool):
    norms, conds = [], []
    for i in range(1000):
        norm, cond = init(network, l2_coef=l2_coef, do_orth=do_orth)
        norms.append(norm)
        conds.append(cond)

    normal_estimator = np.random.lognormal(np.mean(np.log(norms)), np.std(np.log(norms)), len(norms))
    return norms, conds, normal_estimator


def plot_quality_complete_minimization_pb(nb_clients: int, nb_samples: int, dim: int, rank_S: int, latent_dim: int,
                                    noise: int, l2_coef: int):

    network = Network(nb_clients, nb_samples, dim, rank_S, latent_dim, noise=noise)
    plt.close()

    smart_norms, smart_conds, smart_estimator = compute_error(network, smart_init_pb, l2_coef, False)
    power_norms, power_conds, power_estimator = compute_error(network, power_init_pb, l2_coef, False)
    smart_norms, power_norms = np.sort(smart_norms)[5:-5], np.sort(power_norms)[5:-5]
    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot on the first subplot
    axs[0].plot(np.log10(smart_norms), np.linspace(0, 1, len(smart_norms)), label=r"$\alpha = 0$", lw=2)
    axs[0].plot(np.log10(power_norms), np.linspace(0, 1, len(power_norms)), label=r"$\alpha = 1$", lw=2)

    axs[0].set_xlabel("Log(Relative error)", fontsize=FONTSIZE)
    axs[0].set_ylabel("Cumulative distribution function", fontsize=FONTSIZE)
    axs[0].legend(fontsize=FONTSIZE)
    axs[0].set_title("Without orthogonalisation")

    smart_norms, smart_conds, smart_estimator = compute_error(network, smart_init_pb, l2_coef, True)
    power_norms, power_conds, power_estimator = compute_error(network, power_init_pb, l2_coef, True)
    smart_norms, power_norms = np.sort(smart_norms)[5:-5], np.sort(power_norms)[5:-5]
    axs[1].plot(np.log10(smart_norms), np.linspace(0, 1, len(smart_norms)), label="smart init.", lw=2)
    axs[1].plot(np.log10(power_norms), np.linspace(0, 1, len(power_norms)), label="power init.", lw=2)
    # axs[1].set_xscale("log")
    axs[1].set_xlabel("Log(Relative error)", fontsize=FONTSIZE)
    axs[1].legend(fontsize=FONTSIZE)
    axs[1].set_title("With orthogonalisation")

    # Adjust layout to prevent clipping
    plt.tight_layout()

    title = f"../../pictures/distribution_errors_N{network.nb_clients}_d{network.dim}_r{network.plunging_dimension}_U"
    if noise != 0:
        title += f"_eps{noise}"
    if l2_coef != 0:
        title += f"_ridge{l2_coef}"
    plt.savefig(f"{title}.pdf", dpi=600, bbox_inches='tight')


if __name__ == '__main__':

    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 5, 0, 0)
    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 5, 0, 10 ** -9)
    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 6, 0, 0)
    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 6, 0, 10 ** -9)
    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 6, 10**-9, 0)
    plot_quality_complete_minimization_pb(NB_CLIENTS, 100, 100, 5, 6, 10**-9, 10 ** -9)