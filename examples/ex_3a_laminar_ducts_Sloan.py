import mhdtools
import mhdtools.analytic
import numpy as np
import matplotlib.pyplot as plt

# Create a grid

a = 1  # height
b = 1  # width
N_x = 501  # num points in width
N_y = 501  # num points in height
t_w = 0.1  # Hartmann wall thickness

x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a + t_w, b)

# Define Hunt flow (finite conductivity walls) case
# Based on HuntII case (side walls are always perfectly insulating)
# Using Sloan 66 solution for thick walls
# Assumed to be flow along z in a square duct

Ha = 100  # Hartmann number
truncation = 70  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity_f = 1  # Fluid conductivity
conductivity_w = 1  # Solid conductivity
permeability = 1  # Fluid permeability
# average_velocity = 1  # Average flow velocity
pressure_gradient = (
    1  # Pressure gradient (can't currently set based on flow velocity)
)

# Create the case
sloan_case = mhdtools.analytic.Sloan(
    Ha,
    a,
    b,
    t_w,
    truncation,
    x,
    y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
)

# Solve the case
sloan_case.analytic_solve()

# Constrain average velocity

# sloan_case.set_scaled_average_velocity(average_velocity)
sloan_case.set_scaled_pressure_grad(pressure_gradient)

# Calculate scaled fields
sloan_case.calculate_scaled_solution()
sloan_case_uz = sloan_case.scaled_velocity_z
sloan_case_bz = sloan_case.scaled_B_field_z
sloan_case_K = sloan_case.scaled_pressure_grad

# sloan_case_uz = sloan_case.w
# sloan_case_bz = sloan_case.B

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(x, y, sloan_case_uz, cmap="coolwarm", shading="gouraud")
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_3a_laminar_ducts_HuntII_Sloan.png")
print("Pressure Drop K = %f Pa/m" % sloan_case_K)
