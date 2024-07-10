"""
Created by Constantin Philippenko, 15th January 2024.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.Network import Network


class AbstractAlgorithm(ABC):

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str) -> None:
        self.network = network
        self.init_type = init_type
        self.nb_clients = network.nb_clients
        self.nb_epoch = nb_epoch
        self.rho = rho

        self.errors = []

        # STEP-SIZE
        self.__initialization__()

        # REGULARIZATION
        self.l1_coef = 0
        self.l2_coef = 0
        self.nuc_coef = 0

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __initialization__(self):
        pass

    @abstractmethod
    def __epoch_update__(self, it: int):
        pass

    def __F__(self):
        return np.sum([client.loss(client.U, client.V, self.l1_coef, self.l2_coef, self.nuc_coef) for client in self.network.clients])

    def run(self):
        self.errors.append(self.__F__())
        for i in range(self.nb_epoch):
            self.__epoch_update__(i)
        return self.errors