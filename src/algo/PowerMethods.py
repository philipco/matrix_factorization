import numpy as np

from src.Client import Network
from src.MatrixUtilities import orth
from src.algo.AbstractAlgorithm import AbstractAlgorithm
from src.algo.MFInitialization import random_MF_initialization, random_power_iteration


class DistributedPowerMethod(AbstractAlgorithm):

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str, local_epoch: int) -> None:
        super().__init__(network, nb_epoch, rho, init_type)
        self.local_epoch = local_epoch

    def name(self):
        return "LocalPower"
    def __initialization__(self):
        self.sigma_min, self.sigma_max = random_power_iteration(self.network)
        for client in self.network.clients:
            client.set_U(orth(client.U))
            client.set_V(orth(client.V))

    def __epoch_update__(self):
        for k in range(self.local_epoch):
            for client in self.network.clients:
                 client.local_power_iteration()
            self.errors.append(self.__F__())
        # Aggregation on non-orthonomalized vectors V.
        n = np.sum([client.nb_samples for client in self.network.clients])
        V = np.sum([client.V * client.nb_samples / n for client in self.network.clients], axis=0)
        # Orthogonalisation of V.
        V = orth(V)
        for client in self.network.clients:
            client.set_V(V)

