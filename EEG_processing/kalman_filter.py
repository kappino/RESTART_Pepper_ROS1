import numpy as np

class KalmanFilter:
    """Applies Kalman filtering to smooth out noisy gyroscope data."""
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance  # Noise in the system
        self.measurement_variance = measurement_variance  # Measurement noise
        self.estimate = 0  # Initial estimate
        self.estimate_error = 1  # Initial estimate error
        self.kalman_gain = 0  # Kalman gain

    def update(self, measurement):
        """Updates the filter with a new measurement."""
        # Prediction update
        self.estimate_error += self.process_variance
        # Calculate Kalman gain
        self.kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        # Update estimate with new measurement
        self.estimate = self.estimate + self.kalman_gain * (measurement - self.estimate)
        # Update estimate error
        self.estimate_error = (1 - self.kalman_gain) * self.estimate_error
        return self.estimate
