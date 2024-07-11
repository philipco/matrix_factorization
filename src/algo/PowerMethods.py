import numpy as np
import scipy

from src.Network import Network
from src.algo.AbstractAlgorithm import AbstractAlgorithm
from src.algo.MFInitialization import random_power_iteration


class DistributedPowerMethod(AbstractAlgorithm):
    """Implement the distributed power method as presented in our paper."""

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str, local_epoch: int) -> None:
        super().__init__(network, nb_epoch, rho, init_type)
        self.local_epoch = local_epoch

    def name(self):
        return "LocalPower"

    def __initialization__(self):
        self.sigma_min, self.sigma_max = random_power_iteration(self.network)

    def __epoch_update__(self):
        for k in range(self.local_epoch):
            for client in self.network.clients:
                 client.local_power_iteration()
            self.errors.append(self.__F__())
        # Aggregation on non-orthonomalized vectors V.
        n = np.sum([client.nb_samples for client in self.network.clients])
        V = np.sum([client.V * client.nb_samples / n for client in self.network.clients], axis=0)
        # Orthogonalisation of V.
        V = scipy.linalg.orth(V, rcond=0)
        for client in self.network.clients:
            client.set_V(V)

