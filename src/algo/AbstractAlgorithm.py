"""
Created by Constantin Philippenko, 15th January 2024.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.Network import Network


class AbstractAlgorithm(ABC):
    """Abstract class to define any algorithm to do matrix factorisation."""

    def __init__(self, network: Network, nb_epoch: int) -> None:
        self.network = network
        self.nb_clients = network.nb_clients
        self.nb_epoch = nb_epoch

        self.errors = []

        # REGULARIZATION
        self.l1_coef = 0
        self.l2_coef = 0
        self.nuc_coef = 0

    @abstractmethod
    def name(self):
        """"Name of the algorithm."""
        pass

    @abstractmethod
    def __initialization__(self):
        """Initialization of U,V, the factorizing matrices."""
        pass

    @abstractmethod
    def __epoch_update__(self, it: int):
        """Update during one epoch."""
        pass

    def __F__(self):
        """Global objective function that we want to minimize."""
        return np.sum([client.loss(client.U, client.V, self.l1_coef, self.l2_coef, self.nuc_coef) for client in self.network.clients])

    def run(self, eps=None):
        """Run the algorithm of matrix factorisation."""
        self.errors.append(self.__F__())
        if eps is None:
            for i in range(self.nb_epoch):
                self.__epoch_update__(i)
        else:
            i = 0
            while self.errors[-1] > eps and i < 1000:
                self.__epoch_update__(i)
                i+=1
            if i == 1000:
                print("Warning: requires more than 1000 iterations.")
            else:
                print(f"Number of iterations: {i}")
        return self.errors