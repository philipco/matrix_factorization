"""
Created by Constantin Philippenko, 18th January 2024.
"""

import numpy as np
import scipy

from src.algo.AbstractAlgorithm import AbstractAlgorithm
from src.algo.MFInitialization import random_power_iteration


class LocalPower(AbstractAlgorithm):
    """Implement the distributed power method as presented in our paper."""

    def name(self):
        return "LocalPower"

    def __initialization__(self):
        self.sigma_min, self.sigma_max = random_power_iteration(self.network)

    def __epoch_update__(self, i: int):
        self.__agregating__()
        for client in self.network.clients:
             client.local_power_iteration()
        self.__agregating__()
        self.errors.append(self.__F__())

    def __agregating__(self):
        # Aggregation on non-orthonomalized vectors V.
        n = np.sum([client.nb_samples for client in self.network.clients])
        V = np.sum([client.V * client.nb_samples / n for client in self.network.clients], axis=0)
        # Orthogonalisation of V.
        V = scipy.linalg.orth(V, rcond=0)
        for client in self.network.clients:
            client.V = V
            client.U = client.S @ V

