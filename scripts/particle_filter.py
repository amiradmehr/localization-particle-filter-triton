#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseArray, Pose
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import random
import numpy as np
import math

from sensor_model import SensorModel
from motion_model import MotionModel


class Particles:
    def __init__(self, num_particles: int, theta: float = 0.0):
        self.num_particles = num_particles
        self.particles = []
        self.theta = theta

        rospy.init_node('particles', anonymous=True)

        # Publisher for PoseArray messages
        self.particle_pub = rospy.Publisher("particles", PoseArray, queue_size=10)
        self.particles_poses = np.zeros((self.num_particles, 3))

        self.rate = rospy.Rate(1)  # Publishing rate

    def generate_random_particles(self):
        """
        Generate random particles for demonstration purposes
        """
        particles_msg = PoseArray()
        particles_msg.header.stamp = rospy.Time.now()
        particles_msg.header.frame_id = "map"  # Set the frame ID to match your map frame

        for _ in range(self.num_particles):
            particle = Pose()
            particle.position.x = random.uniform(-7, 7)  # Set the range of x coordinates
            particle.position.y = random.uniform(-5, 5)  # Set the range of y coordinates
            particle.position.z = 0

            theta = random.uniform(-math.pi, math.pi)

            euler_angles = [0, 0, theta]
            # Generate a random quaternion
            quaternion = quaternion_from_euler(euler_angles[0], euler_angles[1], euler_angles[2])

            particle.orientation.x = quaternion[0]
            particle.orientation.y = quaternion[1]
            particle.orientation.z = quaternion[2]
            particle.orientation.w = quaternion[3]

            particles_msg.poses.append(particle)

        # store particles poses x,y,theta in numpy array: self.particles_poses
        for i, particle in enumerate(particles_msg.poses):
            self.particles_poses[i, 0] = particle.position.x
            self.particles_poses[i, 1] = particle.position.y
            self.particles_poses[i, 2] = math.atan2(2 * (particle.orientation.w * particle.orientation.z + particle.orientation.x * particle.orientation.y), 1 - 2 * (particle.orientation.y**2 + particle.orientation.z**2))

        print(self.particles_poses)


        # Publish the PoseArray message
        self.particle_pub.publish(particles_msg)


    def publish_particles(self, particles_msg: PoseArray):
        """
        Publish the particles as a PoseArray message
        """
        particles_msg = PoseArray()
        particles_msg.header.stamp = rospy.Time.now()
        particles_msg.header.frame_id = "map"  # Set the frame ID to match your map frame

        for particle in self.particles:
            particles_msg.poses.append(particle)

        # Publish the PoseArray message
        self.particle_pub.publish(particles_msg)
    
    def run(self):
        """
        Main loop to continuously publish particles
        """
        while not rospy.is_shutdown():
            self.generate_random_particles()
            self.rate.sleep()





class ParticleFilter:
    def __init__(self, motion_model: MotionModel, sensor_model: SensorModel, particles: Particles,
                 num_particles: int = 250,):
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.num_particles = num_particles
        self.particles = particles

        self.particles = PoseArray()

    def particle_filter(self, X_t_minus_1:list , u_t: np.ndarray, z_t: np.ndarray) -> list:
        """
        Particle filter algorithm
        :param X_t_minus_1: previous particles
        :param u_t: control input
        :param z_t: sensor measurements
        :return: updated particles
        """

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
            i = np.random.choice(range(self.num_particles), p=weights)
            X_t.append(X_bar_t[i][0])

        return X_t
    

    def convert_particles_to_poses(self, X_t: list) -> PoseArray:
        """
        Convert the particles to Pose objects
        :param X_t: particles
        :return: particles as Pose objects
        """
        particles = PoseArray()
        particles.header.stamp = rospy.Time.now()
        particles.header.frame_id = "map"

        for particle in X_t:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0

            euler_angles = [0, 0, particle[2]]
            quaternion = quaternion_from_euler(euler_angles[0], euler_angles[1], euler_angles[2])

            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            particles.poses.append(pose)

        return particles
    

    def publish_particles(self, particles_msg: PoseArray):
        """
        Publish the particles as a PoseArray message
        """
        particles_msg = PoseArray()
        particles_msg.header.stamp = rospy.Time.now()
        particles_msg.header.frame_id = "map"  # Set the frame ID to match your map frame

        for particle in self.particles:
            particles_msg.poses.append(particle)

        # Publish the PoseArray message
        self.particle_pub.publish(particles_msg)
    

class Robot:
    def __init__(self):
        rospy.init_node('odom_reader', anonymous=True)

        rospy.Rate(1)

        self.x = 0
        self.y = 0
        self.theta = 0

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
        self.theta = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]


def main():
    robot = Robot()


    while not rospy.is_shutdown(): 
        # read robot's lidar data
        ranges = robot.ranges
        # read robot's odometry data
        x = robot.x
        y = robot.y
        theta = robot.theta


        print(f"x: {x}, y: {y}, theta: {theta}")
        print(f"ranges: {ranges}")

        rospy.sleep(1)


def test_arrow():
    rospy.init_node("particle_filter")

    # Create a publisher for the PoseArray messages
    particle_pub = rospy.Publisher("particles", PoseArray, queue_size=10)

    rate = rospy.Rate(1)  # Set the publishing rate

    while not rospy.is_shutdown():
        # Generate random particles (for demonstration purposes)
        num_particles = 10000
        particles = PoseArray()
        particles.header.stamp = rospy.Time.now()
        particles.header.frame_id = "map"  # Set the frame ID to match your map frame

        for _ in range(num_particles):
            particle = Pose()
            particle.position.x = random.uniform(-7, 7)  # Set the range of x coordinates
            particle.position.y = random.uniform(-5, 5)  # Set the range of y coordinates
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
    # num_particles = 2  # Number of particles
    # particles_manager = Particles(num_particles)
    # particles_manager.run()
    main()