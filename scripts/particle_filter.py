#!/usr/bin/env python3

import rospy

import numpy as np
import math

from sensor_model import SensorModel
from motion_model import MotionModel



class ParticleFilter:
    def __init__(self, num_particles: int, z_hit: float, z_max: float = 9.2, z_random: float = 0, sigma_hit: float = 1):
        self.num_particles = num_particles
        self.z_hit = z_hit
        self.z_max = z_max
        self.z_random = z_random
        self.sigma_hit = sigma_hit

        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.sensor_model = SensorModel(z_hit, z_max, z_random, sigma_hit)
        self.motion_model = MotionModel()

    def initialize_particles(self, initial_pose: np.ndarray, initial_cov: np.ndarray):
        """
        Initialize the particles using the initial pose and covariance
        :param initial_pose: initial pose of the robot [x, y, theta]. size 1x3
        :param initial_cov: initial covariance of the robot [cov_x, cov_y, cov_theta]. size 1x3
        """
        self.particles = np.random.multivariate_normal(initial_pose, initial_cov, self.num_particles)

    def predict(self, u: np.ndarray, delta_t: float):
        """
        Predict the particles using the motion model
        :param u: control input [v, w]. size 1x2
        :param delta_t: time step
        """
        for i in range(self.num_particles):
            self.particles[i] = self.motion_model.sample_motion_model(u, self.particles[i], delta_t)

    def update(self, z_t: np.ndarray):
        """
        Update the particle weights using the sensor model
        :param z_t: range finder measurements. size 1xN where N is the number of range finder beams
        """
        for i in range(self.num_particles):
            self.weights[i] = self.sensor_model.likelihood_field_range_finder_model(z_t, self.particles[i])

    def resample(self):
        """
        Resample the particles using the weights
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

    def get_particle(self):
        """
        Get the particle with the highest weight
        :return: particle with the highest weight. size 1x3
        """
        max_weight = np.argmax(self.weights)


