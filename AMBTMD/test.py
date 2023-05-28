# import numpy as np

# # Given accelerometer data
# time = [0.026287385, 0.036297, 0.046306846, 0.056286154, 0.066295769, 0.076305616, 0.086284923,
#         0.096294539, 0.106304385, 0.116283693, 0.126293308, 0.136303154, 0.146282462, 0.156292077,
#         0.166301924, 0.176281231, 0.186290847, 0.196300693, 0.206280001, 0.216289847, 0.226299462,
#         0.23627877, 0.246288616, 0.256298231, 0.266277539, 0.276287385, 0.286297001, 0.296306847, 0.306286155]

# # Calculate time interval
# time_interval = np.diff(time)

# # Print the time interval values
# print(time_interval)

import numpy as np

acceleration = [-0.098650932312012, -0.537377834320068, -0.776372909545898, -0.63924503326416, -0.429503440856934, -
                0.203366756439209, 0.007119655609131, 0.192930221557617, 0.141373634338379]  # Sample accelerometer data
time_interval = 0.001


def calculate_distance(acceleration, time_interval):
    # Step 1: Preprocess the data (if necessary)

    # Step 2: Integrate acceleration to calculate velocity
    velocity = np.cumsum(acceleration) * time_interval

    # Step 3: Integrate velocity to calculate position
    position = np.cumsum(velocity) * time_interval

    # Step 4: Convert position to kilometers
    # Assuming position values are in meters, convert to kilometers
    distance = position / 1000

    return distance


# Example usage:
acceleration = [1.2, 0.8, 1.5, 1.1, 0.9]  # Sample accelerometer data
# Time interval between acceleration measurements (in seconds)
time_interval = 0.1

distance = calculate_distance(acceleration, time_interval)
print(f"Estimated distance traveled: {distance} kilometers")
