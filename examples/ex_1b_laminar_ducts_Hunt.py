import mhdtools
import mhdtools.analytic
import numpy as np
import matplotlib.pyplot as plt

# Create a grid

a = 1  # height
b = 1  # width
N_x = 501  # num points in width
N_y = 501  # num points in height

x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a, b)

# Define Hunt flow (finite conductivity walls) case
# Based on HuntII case (side walls are always perfectly insulating)
# Assumed to be flow along z in a square duct

Ha = 100  # Hartmann number
sigma_w = 1  # Solid conductivity
sigma = 1  # Fluid conductivity
t_w = 0.1  # Hartmann wall thickness
dB = (t_w * sigma_w) / (a * sigma)  # Hartmann wall conduction ratio
num_k_points = 200  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity = 1  # Fluid conductivity
permeability = 1  # Fluid permeability
average_velocity = 1  # Average flow velocity

# Create the case
hunt_case = mhdtools.analytic.HuntII(
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
hunt_case.analytic_solve()

# Constrain average velocity

hunt_case.set_scaled_average_velocity(average_velocity)

# Calculate scaled fields
hunt_case.calculate_scaled_solution()
hunt_case_uz = hunt_case.scaled_velocity_z
hunt_case_bz = hunt_case.scaled_B_field_z
hunt_case_K = hunt_case.scaled_pressure_drop

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(x, y, hunt_case_uz, cmap="coolwarm", shading="gouraud")
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_1b_laminar_ducts_Hunt.png")
print("Pressure Drop K = %f Pa/m" % hunt_case_K)
