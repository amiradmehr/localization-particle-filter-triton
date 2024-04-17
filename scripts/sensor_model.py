# /usr/bin/env python3

import math
import numpy as np
from generate_likelihood_field import LikelihoodField
from scipy.stats import norm


class SensorModel:
    def __init__(self, z_hit: float = 1, z_max: float = 8, z_random: float = 0, sigma_hit: float = 1):
        self.z_hit = z_hit
        self.z_max = z_max
        self.z_random = z_random
        self.sigma_hit = sigma_hit

        self.lf = LikelihoodField()
        self.distance_matrix = self.lf.compute_lookup_table(self.lf.map_path)

    def likelihood_field_range_finder_model(self, z_t: np.ndarray, x_t: np.ndarray) -> float:
        """
        Compute the likelihood of the particle pose given the range finder measurements
        :param z_t: range finder measurements. size 1xN where N is the number of range finder beams
        :param x_t: particle pose [x, y, theta]. size 1x3
        :return: likelihood of the particle pose given the range finder measurements size 1x1 
        """

        x_sensor = np.zeros(z_t.shape[0])
        y_sensor = np.zeros(z_t.shape[0])
        theta_sensor = np.array(list(range(z_t.shape[0])))
        theta_sensor_rad = np.deg2rad(theta_sensor)

        q = 1

        x = x_t[0]
        y = x_t[1]
        theta = x_t[2]

        for i in range(z_t.shape[0]):
                
            if z_t[i] < self.z_max:

                x_zk = x + x_sensor[i]*math.cos(theta) - y_sensor[i]*math.sin(theta) + z_t[i]*math.cos(theta + theta_sensor_rad[i])
                y_zk = y + y_sensor[i]*math.cos(theta) + x_sensor[i]*math.sin(theta) + z_t[i]*math.sin(theta + theta_sensor_rad[i])

                # x_zk, y_zk should be in the map
                if x_zk < 0 or x_zk >= 9 or y_zk < 0 or y_zk >= 9:
                    dist = self.lf.lookup_table[383, 383]
                else:
                    dist = self.lf.get_likelihood_field(x_zk, y_zk)
                    q = q * (self.z_hit * norm.pdf(dist, loc=0, scale=self.sigma_hit) + self.z_random/self.z_max)
        
        return q


def main():
    z_t = np.random.rand(10)*10
    x_t = np.array([0, 0, 0])

    sm = SensorModel(z_hit=1)
    q = sm.likelihood_field_range_finder_model(z_t, x_t)
    print(q)

if __name__ == "__main__":
    main()