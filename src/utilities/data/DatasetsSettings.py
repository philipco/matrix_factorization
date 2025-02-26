"""
Created by Constantin Philippenko, 29th May 2024.

Settings for the four used datasets.
"""


NB_EPOCHS = {"synth": 200, "mnist": 500, "celeba": 500, "w8a": 500}
RANK_S = {"synth": 5, "mnist": None, "celeba": None, "w8a": None}
LATENT_DIMENSION = {"synth": 5, "mnist": 20, "celeba": 20, "w8a": 20}
NB_CLIENTS = {"synth": 25, "mnist": 10, "celeba": 25, "w8a": 25}
NOISE = {"synth": 10**-6, "mnist": 0, "celeba": 0, "w8a": 0}

EPS = {"synth": -5.5, "mnist": 5.5, "celeba": 4.5, "w8a": 5}