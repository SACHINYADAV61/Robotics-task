import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# Step 1: Sensor Modeling and Data Simulation
# ==================================================

# Parameters
time_steps = 100  # Number of time steps
dt = 0.1  # Time step size
wheel_radius = 0.5  # Wheel radius in meters
sensor_noise_std = 0.1  # Standard deviation of sensor noise

# Ground truth path (straight line + turn)
ground_truth_distance = np.linspace(0, 10, time_steps)  # Linear distance
ground_truth_angle = np.zeros(time_steps)  # Angle starts at 0

# Simulate a turn after 50 time steps
ground_truth_angle[50:] = np.linspace(0, np.pi / 2, time_steps - 50)  # 90-degree turn

# Simulate sensor readings with noise
sensor1 = ground_truth_distance + np.random.normal(0, sensor_noise_std, time_steps)
sensor2 = ground_truth_distance + np.random.normal(0, sensor_noise_std, time_steps)

# ==================================================
# Step 2: Kalman Filter for Sensor Fusion
# ==================================================

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state  # State vector [distance, velocity]
        self.covariance = initial_covariance  # State covariance matrix
        self.process_noise = process_noise  # Process noise covariance
        self.measurement_noise = measurement_noise  # Measurement noise covariance
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.F = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.Q = np.array([[process_noise, 0], [0, process_noise]])  # Process noise matrix
        self.R = np.array([[measurement_noise]])  # Measurement noise matrix

    def predict(self):
        # Predict state and covariance
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(self.H @ self.covariance @ self.H.T + self.R)
        # Update state and covariance
        self.state = self.state + K @ (measurement - self.H @ self.state)
        self.covariance = (np.eye(2) - K @ self.H) @ self.covariance
        return self.state

# Initialize Kalman Filter
initial_state = np.array([0, 0])  # Initial distance and velocity
initial_covariance = np.eye(2)  # Initial covariance
process_noise = 0.01  # Process noise
measurement_noise = 0.1  # Measurement noise

kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

# Apply Kalman Filter to sensor data
filtered_distance = np.zeros(time_steps)
for t in range(time_steps):
    # Predict step
    kf.predict()
    # Update step using the average of the two sensors
    measurement = (sensor1[t] + sensor2[t]) / 2
    filtered_state = kf.update(measurement)
    filtered_distance[t] = filtered_state[0]

# ==================================================
# Step 3: Trailer Model
# ==================================================

# Trailer model parameters
trailer_lag = 0.2  # Trailer lags behind by 0.2 seconds

# Simulate trailer path
trailer_distance = np.zeros(time_steps)
trailer_angle = np.zeros(time_steps)

for t in range(1, time_steps):
    trailer_distance[t] = filtered_distance[t - 1]  # Trailer follows the sprayer with a lag
    trailer_angle[t] = ground_truth_angle[t - 1]  # Trailer follows the sprayer's angle with a lag

# ==================================================
# Step 4: Pose Estimation
# ==================================================

# Pose estimation
x, y = np.zeros(time_steps), np.zeros(time_steps)  # Position
theta = np.zeros(time_steps)  # Orientation

for t in range(1, time_steps):
    theta[t] = theta[t - 1] + (ground_truth_angle[t] - ground_truth_angle[t - 1])  # Update orientation
    x[t] = x[t - 1] + filtered_distance[t] * np.cos(theta[t])  # Update x position
    y[t] = y[t - 1] + filtered_distance[t] * np.sin(theta[t])  # Update y position

# ==================================================
# Step 5: Output & Evaluation
# ==================================================

# Plot estimated distance vs ground truth
plt.figure(figsize=(10, 5))
plt.plot(ground_truth_distance, label="Ground Truth Distance")
plt.plot(filtered_distance, label="Filtered Distance (Kalman Filter)")
plt.xlabel("Time Steps")
plt.ylabel("Distance (m)")
plt.legend()
plt.title("Distance Estimation with Kalman Filter")
plt.show()

# Plot trajectory
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Sprayer Trajectory")
plt.plot(x - trailer_lag * np.cos(theta), y - trailer_lag * np.sin(theta), label="Trailer Trajectory")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.title("Trajectory of Sprayer and Trailer")
plt.show()

# ==================================================
# Step 6: Error Analysis
# ==================================================

# Error analysis
error = np.abs(ground_truth_distance - filtered_distance)
plt.figure(figsize=(10, 5))
plt.plot(error, label="Estimation Error")
plt.xlabel("Time Steps")
plt.ylabel("Error (m)")
plt.legend()
plt.title("Distance Estimation Error with Kalman Filter")
plt.show()

# ==================================================
# Step 7: Documentation
# ==================================================

 
