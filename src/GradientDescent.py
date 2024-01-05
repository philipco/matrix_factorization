"""
Created by Constantin Philippenko, 11th December 2023.
"""
from abc import ABC, abstractmethod
from src.MFInitialization import *


class AbstractGradientDescent(ABC):

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str, l1_coef: int = 0, l2_coef: int = 0,
                 use_momentum: bool = False) -> None:
        self.network = network
        self.init_type = init_type
        self.nb_clients = network.nb_clients
        self.nb_epoch = nb_epoch
        self.rho = rho

        self.sigma_max, self.sigma_min = None, None

        # STEP-SIZE
        self.largest_sv_S = self.__compute_largest_eigenvalues_dataset__()
        self.__descent_initialization__()
        if self.sigma_max is not None:
            self.step_size = 1 / (self.sigma_max**2)
        else:
            self.step_size = 1 / (self.largest_sv_S**2)

        # MOMEMTUM
        self.use_momentum = use_momentum
        if self.sigma_max is not None:
            self.momentum = (np.sqrt(self.sigma_max**2) - np.sqrt(self.sigma_min**2)) / (np.sqrt(self.sigma_max**2) + np.sqrt(self.sigma_min**2))
        else:
            self.momentum = 1 / self.largest_sv_S**2

        # REGULARIZATION
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def variable_optimization(self):
        pass

    def __descent_initialization__(self):
        # np.random.seed(42)
        if self.init_type == "SMART" and self.variable_optimization() != "U":
            self.sigma_min, self.sigma_max = smart_MF_initialization(self.network)
        elif self.init_type == "SMART" and self.variable_optimization() == "U":
            self.sigma_min, self.sigma_max = smart_MF_initialization_for_GD_on_U(self.network)
        elif self.init_type == "BI_SMART" and self.variable_optimization() != "U":
            self.sigma_min, self.sigma_max = bi_smart_MF_initialization(self.network)
        elif self.init_type == "BI_SMART" and self.variable_optimization() == "U":
            self.sigma_min, self.sigma_max = bi_smart_MF_initialization_for_GD_on_U(self.network)
        elif self.init_type == "ORTHO" and self.variable_optimization() != "U":
            self.sigma_min, self.sigma_max = ortho_MF_initialization(self.network)
        elif self.init_type == "ORTHO" and self.variable_optimization() == "U":
            self.sigma_min, self.sigma_max = ortho_MF_initialization_for_GD_on_U(self.network)
        elif self.init_type == "RANDOM":
            random_MF_initialization(self.network)
        else:
            raise ValueError("Unrecognized type of initialization.")
        # self.smallest_sv = self.__compute_smallest_eigenvalues_init__()
        self.largest_sv = self.__compute_largest_eigenvalues_init__()

    @abstractmethod
    def __epoch_update__(self):
        pass

    def __compute_largest_eigenvalues_dataset__(self):
        largest_sv_S = 0
        for client in self.network.clients:
            largest_sv_S += svds(client.S, k=self.network.plunging_dimension - 1, which='LM')[1][-1]
        # print("{0}, {1}:\nDataset largest singular value: {2}".format(self.name(), self.init_type, largest_sv_S))
        return largest_sv_S

    def __compute_largest_eigenvalues_init__(self):
        largest_sv_U = 0
        largest_sv_V = 0
        for client in self.network.clients:
            largest_sv_U += svds(client.U, k=self.network.plunging_dimension - 1, which='LM')[1][-1]
            largest_sv_V += svds(client.V, k=self.network.plunging_dimension - 1, which='LM')[1][-1]
        # print("Largest singular value: ({0}, {1})".format(largest_sv_U, largest_sv_V))
        return (largest_sv_U, largest_sv_V)

    def __compute_smallest_eigenvalues_init__(self):
        smallest_sv_U = 0
        smallest_sv_V = 0
        for client in self.network.clients:
            smallest_sv_U += svds(client.U, k=self.network.plunging_dimension - 1, which='SM')[1][0]
            smallest_sv_V += svds(client.V, k=self.network.plunging_dimension - 1, which='SM')[1][0]
        print("Smallest singular value: ({0}, {1})".format(smallest_sv_U, smallest_sv_V))
        return (smallest_sv_U, smallest_sv_V)

    def __F__(self):
        return np.mean([client.loss() for client in self.network.clients])

    def gradient_descent(self):
        error = [self.__F__()]

        for i in range(self.nb_epoch):
            self.__epoch_update__()
            error.append(self.__F__())
        return error


class GD(AbstractGradientDescent):

    def name(self):
        return "GD"

    def variable_optimization(self):
        return "U,V"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            gradU = self.step_size * client.local_grad_wrt_U(client.U, self.l1_coef, self.l2_coef)
            gradV.append(client.local_grad_wrt_V(client.V, self.l1_coef, self.l2_coef))
            client.U -= self.step_size * gradU
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.sum([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])

class AlternateGD(AbstractGradientDescent):

    def name(self):
        return "Alternate GD"

    def variable_optimization(self):
        return "U,V"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            client.U -= self.step_size * self.step_size * client.local_grad_wrt_U(client.U, self.l1_coef, self.l2_coef)
            gradV.append(client.local_grad_wrt_V(client.V, self.l1_coef, self.l2_coef))
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.sum([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])

class GD_ON_V(AbstractGradientDescent):

    def name(self):
        return "GD on V"

    def variable_optimization(self):
        return "V"

    def __epoch_update__(self):
        gradV = []
        Vold = []
        for client in self.network.clients:
            gradV.append(client.local_grad_wrt_V(client.V, self.l1_coef, self.l2_coef))
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.sum([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])


class GD_ON_U(AbstractGradientDescent):

    def name(self):
        return "GD on U"

    def variable_optimization(self):
        return "U"

    def __epoch_update__(self):
        for client in self.network.clients:
            if self.use_momentum:
                client.U_past = client.U
                client.U = client.U_half - self.step_size * client.local_grad_wrt_U(client.U_half, self.l1_coef,
                                                                                    self.l2_coef)
                client.U_half = client.U + self.momentum * (client.U - client.U_past)
            else:
                client.U = client.U - self.step_size * client.local_grad_wrt_U(client.U, self.l1_coef, self.l2_coef)

