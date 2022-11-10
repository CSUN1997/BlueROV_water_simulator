import numpy as np
import random
import matplotlib.pyplot as plt


class Current(object):
    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.integral = 0.1

    def get_current_derivative(self):
        return np.random.normal(0, 0.1)

    def current_integrator(self, current_derivative):
        self.integral += self.delta_t * current_derivative
        return self.integral

    def limit(self, signal_val):
        return np.clip(signal_val, 0, 0.2)

    def get_vc(self):
        current_derivative = self.get_current_derivative()
        signal_val = self.current_integrator(current_derivative)
        return self.limit(signal_val)

    def get_current_vector(self, beta, psi):
        vc = self.get_vc()
        vc_b = np.asarray([vc * np.cos(beta - psi), vc * np.sin(beta - psi), 0])
        return vc_b


class ROVHorizontalModel(object):
    @staticmethod
    def rov_hm(m, Iz, rg, HydroP, T_Config, Thrust, vc_b, eta_i, vr_b):
        xg = rg[0]
        yg = rg[1]
        Xu = HydroP[0]
        Yv = HydroP[1]
        Nr = HydroP[2]
        Xdu = HydroP[3]
        Ydv = HydroP[4]
        Ndr = HydroP[5]
        Lh = T_Config[0]
        Wh = T_Config[1]
        gamma1 = T_Config[2]
        gamma2 = T_Config[3]
        gamma3 = T_Config[4]
        gamma4 = T_Config[5]
        psi = eta_i[2]
        ur = vr_b[0]
        vr = vr_b[1]
        r = vr_b[2]
        J_bi = np.asarray([[np.cos(psi), -np.sin(psi), 0],
                           [np.sin(psi), np.cos(psi), 0],
                           [0, 0, 1]])
        deta_i = J_bi @ (vr_b + vc_b)
        M = np.asarray([[m - Xdu, 0, -m * yg],
                        [0, m - Ydv, m * xg],
                        [-m * yg, m * xg, Iz - Ndr]])
        C = np.asarray([[0, 0, -(m - Ydv) * vr - m * xg * r],
                        [0, 0, -m * yg * r + (m - Xdu) * ur],
                        [(m - Ydv) * vr + m * xg * r, m * yg * r - (m - Xdu) * ur, 0]])
        D = -np.asarray([[Xu, 0, 0],
                         [0, Yv, 0],
                         [0, 0, Nr]])
        B = np.asarray([[np.cos(gamma1), np.cos(gamma2), np.cos(gamma3), np.cos(gamma4)],
                        [np.sin(gamma1), np.sin(gamma2), np.sin(gamma3), np.sin(gamma4)],
                        [Lh * np.sin(gamma1) - Wh * np.cos(gamma1), Lh * np.sin(gamma2) + Wh * np.cos(gamma2),
                         -Lh * np.sin(gamma3) + Wh * np.cos(gamma3), -Lh * np.sin(gamma4) - Wh * np.cos(gamma4)]])
        dvr_b = np.linalg.inv(M) @ (B @ Thrust - C @ vr_b - D @ vr_b)
        return deta_i, dvr_b


class ROVModelCurrent(object):
    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.current = Current(delta_t)

        self.beta = 0
        self.psi = 0
        self.m = 10
        self.Iz = 2
        self.rg = np.asarray([-0.001, 0.01]).T
        self.HydroP = np.asarray([-7.2, -7.7, -3, -2.9, -3, -0.33]).T
        self.T_Config = np.asarray([0.145, 0.1, -np.pi / 4, np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]).T
        self.Thrust = 0.25 * np.asarray([50, 50, -40, -40]).T
        # self.Thrust = np.zeros(4).T
        self.v0_b = np.zeros(3).T

        self.eta_i = np.zeros(3).T
        self.vc = self.current.get_vc()
        self.vc_b = self.current.get_current_vector(self.beta, self.psi)
        # self.vc = np.zeros(3).T
        self.vr_b = self.v0_b - self.vc_b

    def rollout(self):
        # vc_b = np.zeros(3).T
        deta_i, dvr_b = ROVHorizontalModel.rov_hm(self.m, self.Iz, self.rg, self.HydroP, self.T_Config,
                                                  self.Thrust, self.vc_b, self.eta_i, self.vr_b)
        self.vr_b += self.delta_t * dvr_b
        self.eta_i += self.delta_t * deta_i
        # self.psi = np.rad2deg(self.eta_i[-1])
        # self.psi = self.eta_i[-1]
        self.psi = self.eta_i[-1] * 57.3
        self.vc = self.current.get_vc()
        self.vc_b = self.current.get_current_vector(self.beta, self.psi)

    def set_thrust_force(self, thrust_force):
        assert len(thrust_force) == 4
        self.Thrust = thrust_force.T


if __name__ == "__main__":
    rov_model = ROVModelCurrent(0.1)
    x = []
    y = []
    psi = []
    for i in range(50):
        rov_model.rollout()
        x.append(rov_model.eta_i[0])
        y.append(rov_model.eta_i[1])
        psi.append(rov_model.eta_i[2])
    x, y, psi = np.asarray(x), np.asarray(y), np.asarray(psi)
    dx = np.cos(psi) * 1
    dy = np.sin(psi) * 1
    # plt.plot(x, label='x')
    # plt.plot(y, label='y')
    # plt.plot(psi, label='psi')
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.arrow(x[i], y[i], dx[i], dy[i])
    # plt.legend()
    plt.show()
    # plt.plot(sig)
    # plt.show()
