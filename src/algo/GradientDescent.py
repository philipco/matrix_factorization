"""
Created by Constantin Philippenko, 11th December 2023.
"""
from abc import abstractmethod

import scipy
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, SGDRegressor

from src.algo.AbstractAlgorithm import AbstractAlgorithm
from src.algo.MFInitialization import *

class AbstractGradientDescent(AbstractAlgorithm):

    def __init__(self, network: Network, nb_epoch: int, rho: int, init_type: str, l1_coef: int = 0, l2_coef: int = 0,
                 nuc_coef: int = 0, use_momentum: bool = False) -> None:
        super().__init__(network, nb_epoch, rho, init_type)

        self.sigma_max, self.sigma_min = None, None

        # STEP-SIZE
        self.__initialization__()
        self.__compute_step_size__()

        # MOMEMTUM
        self.use_momentum = use_momentum
        if self.sigma_max is not None:
            kappa = self.sigma_max ** 2 / self.sigma_min ** 2
            self.momentum = 0.95 #(np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)  # 1 / self.largest_sv_S**2
            # self.momentum = (np.sqrt(self.sigma_max**2) - np.sqrt(self.sigma_min**2)) / (np.sqrt(self.sigma_max**2) + np.sqrt(self.sigma_min**2))
        else:
            self.momentum = 1 / self.largest_sv_S**2

        # REGULARIZATION
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.nuc_coef = nuc_coef

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def variable_optimization(self):
        pass

    def __compute_step_size__(self):
        self.step_size = 1 / (self.sigma_max**2)

    def __initialization__(self):
        # np.random.seed(42)
        if self.init_type == "SMART" and self.variable_optimization() == "U":
            self.network.power = 0
            self.sigma_min, self.sigma_max = smart_MF_initialization_for_GD_on_U(self.network)
        elif self.init_type == "POWER" and self.variable_optimization() == "U":
            self.network.power = 1
            self.sigma_min, self.sigma_max = smart_MF_initialization_for_GD_on_U(self.network)
        elif self.init_type == "RANDOM":
            self.sigma_min, self.sigma_max = random_MF_initialization(self.network)
        else:
            raise ValueError("Unrecognized type of initialization.")

    @abstractmethod
    def __epoch_update__(self):
        pass

    def __compute_largest_eigenvalues_init__(self):
        largest_sv_U = 0
        largest_sv_V = 0
        for client in self.network.clients:
            largest_sv_U += svds(client.U, k=self.network.plunging_dimension - 1, which='LM')[1][-1]
            largest_sv_V += svds(client.V, k=self.network.plunging_dimension - 1, which='LM')[1][-1]
        return (largest_sv_U, largest_sv_V)

    def __compute_smallest_eigenvalues_init__(self):
        smallest_sv_U = 0
        smallest_sv_V = 0
        for client in self.network.clients:
            smallest_sv_U += svds(client.U, k=self.network.plunging_dimension - 1, which='SM')[1][0]
            smallest_sv_V += svds(client.V, k=self.network.plunging_dimension - 1, which='SM')[1][0]
        print("Smallest singular value: ({0}, {1})".format(smallest_sv_U, smallest_sv_V))
        return (smallest_sv_U, smallest_sv_V)

    def elastic_net(self, l1_coef, l2_coef):
        error = 0
        for client in self.network.clients:
            if l1_coef == 0 and l2_coef == 0:
                regr = SGDRegressor(fit_intercept=False, penalty=None, eta0=self.step_size, learning_rate="constant",
                                    max_iter=self.nb_epoch)
            elif l1_coef == 0:
                regr = Ridge(fit_intercept=False, alpha=l2_coef / 2, solver="sparse_cg", max_iter=self.nb_epoch)
            else:
                regr = ElasticNet(fit_intercept=False, alpha=l1_coef + l2_coef, l1_ratio=l1_coef / (l1_coef + l2_coef))

            regr.fit(client.V, client.S.T)
            client.U = regr.coef_
            error += client.loss(client.U, client.V, l1_coef, l2_coef, 0)
        return error

    def compute_exact_solution(self, l1_coef, l2_coef, nuc_coef):
        error = 0
        if self.variable_optimization() == "U":
            V = self.network.clients[0].V_0  # Clients share the same V.
            VV = V.T @ V + + l2_coef * np.identity(V.shape[1])
            try:
                VVinv = scipy.linalg.pinvh(VV)
            except np.linalg.LinAlgError:
                return None
            for client in self.network.clients:
                SV = client.S @ V
                client.U = SV @ VVinv
                error += client.loss(client.U, client.V, 0, l2_coef, 0)
        else:
            sum_S_Ui = np.sum([client.S.T @ client.U_0 for client in self.network.clients], axis=0)
            sum_UU = np.zeros((self.network.plunging_dimension, self.network.plunging_dimension))
            for client in self.network.clients:
                sum_UU += client.U_0.T @ client.U_0
            try:
                sum_UUinv = np.linalg.inv(sum_UU + l2_coef * np.identity((self.network.plunging_dimension)))
            except np.linalg.LinAlgError:
                return 10
            for client in self.network.clients:
                client.V = sum_S_Ui @ sum_UUinv
            error += client.loss(client.U, client.V, 0, l2_coef)
        return error


class GD(AbstractGradientDescent):

    def name(self):
        return "GD"

    def variable_optimization(self):
        return "U,V"

    def __epoch_update__(self, it: int):
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
        self.errors.append(self.__F__())

class AlternateGD(AbstractGradientDescent):

    def name(self):
        return "Alternate GD"

    def variable_optimization(self):
        return "U,V"

    def __epoch_update__(self, it: int):
        gradV = []
        Vold = []
        for client in self.network.clients:
            client.U -= self.step_size * client.local_grad_wrt_U(client.U, self.l1_coef, self.l2_coef)
            gradV.append(client.local_grad_wrt_V(client.V, self.l1_coef, self.l2_coef))
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.sum([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])
        self.errors.append(self.__F__())

class GD_ON_V(AbstractGradientDescent):

    def name(self):
        return "GD on V"

    def variable_optimization(self):
        return "V"

    def __epoch_update__(self, it: int):
        gradV = []
        Vold = []
        for client in self.network.clients:
            gradV.append(client.local_grad_wrt_V(client.V, self.l1_coef, self.l2_coef))
            Vold.append(client.V)
        for client in self.network.clients:
            client.V = ((1 - self.rho) * Vold[client.id]
                        + self.rho * np.sum([self.network.W[client.id - 1, k - 1] * Vold[k] for k in range(self.nb_clients)], axis=0)
                        - self.step_size * gradV[client.id])
        self.errors.append(self.__F__())


class GD_ON_U(AbstractGradientDescent):

    def name(self):
        return "GD on U"

    def variable_optimization(self):
        return "U"

    def __epoch_update__(self, it: int):
        for client in self.network.clients:
            if self.use_momentum:
                client.U_past = client.U
                client.U = client.U_half - self.step_size * client.local_grad_wrt_U(client.U_half, self.l1_coef,
                                                                                    self.l2_coef, self.nuc_coef)
                client.U_half = client.U + (client.U - client.U_past) * it / (it + 3)
            else:
                client.U = client.U - self.step_size * client.local_grad_wrt_U(client.U, self.l1_coef, self.l2_coef,
                                                                               self.nuc_coef)
        self.errors.append(self.__F__())


class DGDLocal(AbstractGradientDescent):

    def name(self):
        return "DGD+Local"

    def variable_optimization(self):
        return "U,V"

    def __epoch_update__(self, it: int):
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
        self.errors.append(self.__F__())
