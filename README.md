# Robotics-task
# Distance and Pose Estimation with Dual Proximity Sensors & Trailer Model
# Project Overview

 This project simulates an autonomous sprayer equipped with two proximity sensors mounted on a wheel. The goal is to estimate the distance traveled and the pose (position and orientation) of the sprayer over time. The simulation incorporates a trailer model to account for pose estimation errors during turns. A Kalman Filter is used to fuse the sensor data and reduce noise for accurate estimation.
# Sensor Modeling:
Two proximity sensors simulate wheel-based odometry with realistic Gaussian noise.
A ground truth path is defined, including straight segments and turns.

# Trailer Model:
A simple trailer model follows the sprayer with a slight lag during turns.

# Kalman Filter:
A Kalman Filter is used to fuse the sensor data and reduce noise for accurate distance and pose estimation.

# Pose Estimation:
The sprayer’s position and orientation are estimated over time, incorporating the trailer model.

# Visualization:
Plots show the estimated distance vs ground truth, trajectory of the sprayer and trailer, and error analysis.
