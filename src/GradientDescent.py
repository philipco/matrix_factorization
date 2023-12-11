"""
Created by Constantin Philippenko, 11th December 2023.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.Client import Client, Network
from src.MFInitialization import smart_MF_initialization, random_MF_initialization, smart_MF_initialization_for_GD_on_U

C, NU = 4, 2
D = C * NU / 9

class AbstractGradientDescent(ABC):

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str) -> None:
        # super().__init__()
        self.network = network
        self.nb_clients = network.nb_clients

        self.step_size = 9 / (4 * C * NU)
        self.nb_epoch = nb_epoch
        self.init_type = init_type
        self.rho = rho

    @abstractmethod
    def name(self):
        pass

    def __descent_initialization__(self):
        if self.init_type == "SMART":
            smart_MF_initialization(self.network, self.step_size, C, D)
        elif self.init_type == "SMART_FOR_GD_ON_U":
            smart_MF_initialization_for_GD_on_U(self.network, self.step_size, C, D)
        elif self.init_type == "RANDOM":
            random_MF_initialization(self.network, self.step_size)
        else:
            raise ValueError("Unrecognized type of initialization.")

    @abstractmethod
    def __epoch_update__(self):
        pass

    def __F__(self):
        return np.mean([client.loss() for client in self.network.clients])

    def gradient_descent(self):
        self.__descent_initialization__()
        error = [self.__F__()]

        for i in range(self.nb_epoch):
            self.__epoch_update__()
            error.append(self.__F__())
        return error


class GD(AbstractGradientDescent):

    def name(self):
        return "GD"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            gradU = self.step_size * client.local_grad_wrt_U()
            gradV.append( client.local_grad_wrt_V())
            client.U -= self.step_size * gradU
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.mean([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])

class AlternateGD(AbstractGradientDescent):

    def name(self):
        return "Alternate GD"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            client.U -= self.step_size * self.step_size * client.local_grad_wrt_U()
            gradV.append( client.local_grad_wrt_V())
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.mean([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])

class GD_ON_V(AbstractGradientDescent):

    def name(self):
        return "GD on V"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            gradV.append(client.local_grad_wrt_V())
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.mean([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])


class GD_ON_U(AbstractGradientDescent):

    def name(self):
        return "EmbarrassinglyParallelGD"

    def __epoch_update__(self):
        for client in self.network.clients:
            client.U = client.U - self.step_size * client.local_grad_wrt_U()