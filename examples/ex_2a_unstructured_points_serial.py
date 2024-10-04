import mhdtools
import mhdtools.analytic
import numpy as np
import matplotlib.pyplot as plt
import time

"""
This example demonstrates how to use the analytic.HuntII class to generate solutions on
a set of scattered points, rather than assuming a structured orthogonal cartesian mesh.

The time taken to calculate the solution is measured for comparison to the same
calculation in the parallel example. For small lists of points, serial is likely faster,
however parallelism helps for longer lists. To test this, try changing the num_repeats
variable in both cases to see how the time changes.
"""

# Set number of times to increase the length of the list for comparing performance
num_repeats = 1

# Define a 2D rectangular channel for the MHD case, based on the Hunt-II case, which
# simplifies to the Shercliff case in the limit of perfectly insulating side walls.
# Assumed to be flow along z in a rectangular duct

Ha = 20  # Hartmann number
a = 1  # height
b = 1  # width
dB = 0  # Hartmann wall conductivity: insulating
num_k_points = 200  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity = 1  # Fluid conductivity
permeability = 1  # Fluid permeability
average_velocity = 1  # Average flow velocity

# Define a list of (x, y) points to probe the solution

points = [(0.0, 0.0), (1.0, 1.0), (0.0, -0.5), (0.957, -0.015)]
base_list_length = len(points)

# Repeat the list of points num_repeats times

points = points * num_repeats

# Begin timing
start = time.time()

# For each point, define the Shercliff flow (insulating walls) case, calculate the
# solution, and append them to a list.

shercliff_case_uz = []
shercliff_case_bz = []

for x, y in points:

    # Create the case
    shercliff_case = mhdtools.analytic.HuntII(
        Ha,
        a,
        b,
        dB,
        num_k_points,
        x,
        y,
        dyn_visc,
        conductivity,
        permeability,
    )

    # Solve the case
    shercliff_case.analytic_solve()

    # Constrain average velocity

    shercliff_case.set_scaled_average_velocity(average_velocity)

    # Calculate scaled fields
    shercliff_case.calculate_scaled_solution()
    shercliff_case_uz.append(shercliff_case.scaled_velocity_z)
    shercliff_case_bz.append(shercliff_case.scaled_B_field_z)

# Stop timing
end = time.time()

if num_repeats > 1:
    print(
        "Note: skipping output for additional repeats of"
        "these points because num_repeats > 1"
    )
for i in range(base_list_length):
    x, y = points[i]
    uz = shercliff_case_uz[i]
    bz = shercliff_case_bz[i]
    print("x=%.3f, y=%.3f, uz=%.3f, bz=%.3f" % (x, y, uz, bz))

print("Solve time: %f s" % (end - start))
