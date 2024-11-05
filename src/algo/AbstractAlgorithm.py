"""
Created by Constantin Philippenko, 15th January 2024.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.Network import Network


class AbstractAlgorithm(ABC):
    """Abstract class to define any algorithm to do matrix factorisation."""

    def __init__(self, network: Network, nb_epoch: int, init_type: str) -> None:
        self.network = network
        self.init_type = init_type
        self.nb_clients = network.nb_clients
        self.nb_epoch = nb_epoch

        self.errors = []

        # STEP-SIZE
        self.__initialization__()

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

    def run(self):
        """Run the algorithm of matrix factorisation."""
        self.errors.append(self.__F__())
        for i in range(self.nb_epoch):
            self.__epoch_update__(i)
        return self.errors