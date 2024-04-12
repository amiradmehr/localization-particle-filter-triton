# /usr/bin/env python3

import numpy as np

class MotionModel:
    def __init__(self, alpha: np.ndarray = np.array([0.1, 0.1, 0.1, 0.1])):
        self.alpha = alpha


    def sample_motion_model_odometry(self, u: np.ndarray, x_t_1: np.ndarray) -> np.ndarray:
        """
        Sample motion model odometry
        :param u: control input [x_bar_t_1, x_bar_t] size 2x3
        :param x_t_1: previous state [x, y, theta] size 1x3
        :param alpha: motion model parameters size 1x4
        :return: new state [x_prime, y_prime, theta_prime] size 1x3
        """

        x_bar_t_1 = u[0]
        x_bar_t = u[1]

        x_bar =x_bar_t_1[0]
        y_bar = x_bar_t_1[1]
        theta_bar = x_bar_t_1[2]

        x_bar_prime = x_bar_t[0]
        y_bar_prime = x_bar_t[1]
        theta_bar_prime = x_bar_t[2]

        delta_rot1 = np.arctan2(y_bar_prime - y_bar, x_bar_prime - x_bar) - theta_bar
        delta_trans = np.sqrt((x_bar - x_bar_prime)**2 + (y_bar - y_bar_prime)**2)
        delta_rot2 = theta_bar_prime - theta_bar - delta_rot1

        delta_rot1_hat = delta_rot1 - np.random.normal(0, self.alpha[0]*delta_rot1**2 + self.alpha[1]*delta_trans**2)
        delta_trans_hat = delta_trans - np.random.normal(0, self.alpha[2]*delta_trans**2 + self.alpha[3]*delta_rot1**2 + self.alpha[3]*delta_rot2**2)
        delta_rot2_hat = delta_rot2 - np.random.normal(0, self.alpha[0]*delta_rot2**2 + self.alpha[1]*delta_trans**2)


        x = x_t_1[0]
        y = x_t_1[1]
        theta = x_t_1[2]

        x_prime = x + delta_trans_hat*np.cos(theta + delta_rot1_hat)
        y_prime = y + delta_trans_hat*np.sin(theta + delta_rot1_hat)
        theta_prime = theta + delta_rot1_hat + delta_rot2_hat

        x_t = np.array([x_prime, y_prime, theta_prime])

        return x_t


def main():
    u = np.array([[0, 0, 0], [1, 1, 1]])
    x_t_1 = np.array([0, 0, 0])
    alpha = np.array([0.01, 0.01, 0.01, 0.01])

    motion_model = MotionModel(alpha)

    x_t = motion_model.sample_motion_model_odometry(u, x_t_1)
    print(x_t)

if __name__ == "__main__":
    main()




