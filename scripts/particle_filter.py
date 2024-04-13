#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseArray, Pose
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Header, ColorRGBA

import random
import numpy as np
import math

from sensor_model import SensorModel
from motion_model import MotionModel


class ParticleFilter:
    def __init__(self, motion_model: MotionModel, sensor_model: SensorModel, num_particles: int = 250):
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.num_particles = num_particles

        self.particles = np.zeros((self.num_particles, 3))

    def particle_filter(self, X_t_minus_1, u_t, z_t):
        X_bar_t = []  # Temporary list to store particles and weights
        X_t = []  # List to store resampled particles

        # Step 1: Prediction
        for x_t_minus_1 in X_t_minus_1:
            x_t = self.motion_model.sample_motion_model_odometry(u_t, x_t_minus_1)
            w_t = self.sensor_model.likelihood_field_range_finder_model(z_t, x_t)
            X_bar_t.append((x_t, w_t))

        # Step 2: Update
        weights = [w for _, w in X_bar_t]
        weights = np.array(weights) / np.sum(weights)  # Normalize weights

        # Step 3: Resampling
        for _ in range(self.M):
            i = np.random.choice(range(self.M), p=weights)
            X_t.append(X_bar_t[i][0])

        return X_t

class Particles:
    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.particles = np.zeros((self.num_particles, 3))

        rospy.init_node('particles', anonymous=True)
        rospy.init_node("particle_filter")

        particle_topic = "particles"
        # Create a publisher for the PoseArray messages
        particle_pub = rospy.Publisher(particle_topic, PoseArray, queue_size=10)

        rate = rospy.Rate(1)  # Set the publishing rate

        while not rospy.is_shutdown():
            # Generate random particles (for demonstration purposes)
            num_particles = 1
            particles = PoseArray()
            particles.header.stamp = rospy.Time.now()
            particles.header.frame_id = "map"  # Set the frame ID to match your map frame

            for _ in range(num_particles):
                particle = Pose()
                particle.position.x = random.uniform(-7, 7)  # Set the range of x coordinates
                particle.position.y = random.uniform(-6, 6)  # Set the range of y coordinates
                particle.position.z = 0

                theta = random.uniform(-math.pi, math.pi)

                euler_angles = [0, 0, theta]
                quaternion = euler_from_quaternion(euler_angles)

                particle.orientation.x = quaternion[0]
                particle.orientation.y = quaternion[1]
                particle.orientation.z = quaternion[2]
                particle.orientation.w = quaternion[3]


                particles.poses.append(particle)

            # Publish the PoseArray message
            particle_pub.publish(particles)

            rate.sleep()



class Robot:
    def __init__(self, x: float, y: float, theta: float):

        rospy.init_node('Triton', anonymous=True)
        rospy.init_node('odom_reader', anonymous=True)


        self.ranges = None
        self.lidar_topic = '/scan'
        self.odom_topic = '/odom'
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

    def lidar_callback(self, msg: LaserScan):
        """
        Callback function for the lidar subscriber
        :param msg: LaserScan message
        """
        self.ranges = np.array(msg.ranges)

    def odom_callback(self, msg: Odometry):
        """
        Callback function for the odometry subscriber
        :param msg: Odometry message
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = msg.pose.pose.orientation.z

    def move(self, v: float, w: float, delta_t: float):
        """
        Move the robot using the control inputs
        :param v: linear velocity
        :param w: angular velocity
        :param delta_t: time step
        """
        self.x += v * math.cos(self.theta) * delta_t
        self.y += v * math.sin(self.theta) * delta_t
        self.theta += w * delta_t




def test_arrow():
    rospy.init_node("particle_filter")

    # Create a publisher for the PoseArray messages
    particle_pub = rospy.Publisher("particles", PoseArray, queue_size=10)

    rate = rospy.Rate(1)  # Set the publishing rate

    while not rospy.is_shutdown():
        # Generate random particles (for demonstration purposes)
        num_particles = 100
        particles = PoseArray()
        particles.header.stamp = rospy.Time.now()
        particles.header.frame_id = "map"  # Set the frame ID to match your map frame

        for _ in range(num_particles):
            particle = Pose()
            particle.position.x = random.uniform(-7, 7)  # Set the range of x coordinates
            particle.position.y = random.uniform(-6, 6)  # Set the range of y coordinates
            particle.position.z = 0

            theta = random.uniform(-math.pi, math.pi)
            particle.orientation.x = 0
            particle.orientation.y = 0
            particle.orientation.z = math.sin(theta / 2)

            # Calculate the quaternion w component
            particle.orientation.w = math.cos(theta / 2)

            particles.poses.append(particle)

        # Publish the PoseArray message
        particle_pub.publish(particles)

        rate.sleep()

if __name__ == "__main__":
    test_arrow()