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

# Define Shercliff flow (insulating walls) case
# Based on HuntII case (side walls are always perfectly insulating)
# Assumed to be flow along z in a square duct

Ha = 20  # Hartmann number
dB = 0  # Hartmann wall conductivity: insulating
num_k_points = 200  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity = 1  # Fluid conductivity
permeability = 1  # Fluid permeability
average_velocity = 1  # Average flow velocity

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
shercliff_case_uz = shercliff_case.scaled_velocity_z
shercliff_case_bz = shercliff_case.scaled_B_field_z
shercliff_case_K = shercliff_case.scaled_pressure_drop

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(x, y, shercliff_case_uz, cmap="coolwarm", shading="gouraud")
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_1a_laminar_ducts_Shercliff.png")
print("Pressure Drop K = %f Pa/m" % shercliff_case_K)
