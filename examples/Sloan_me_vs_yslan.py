import mhdtools
import mhdtools.analytic
import time
import mhdtools.yslan
import numpy as np
import matplotlib.pyplot as plt

# Define functions for Hunt and Sloan solutions


def yslan_solution(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
    timing=False,
):
    # Create the grid

    t0 = time.perf_counter()
    x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a, b)

    sol = mhdtools.yslan.Sloan(Ha, conductivity_f, conductivity_w, t_w, a, b)
    #        [Y, X] = np.meshgrid(xx, yy, indexing='xy')
    #        V, H = sol.VH(X, Y, num_k)
    V, H = sol.VH(x, y, truncation, tensor=True)

    v0, _ = sol.VH(0, 0, truncation, tensor=False)
    V /= v0
    H /= v0
    t1 = time.perf_counter() - t0
    if timing: print(f"Yslan: time {t1:.2e}")
    return x, y, V, H


def Sloan_solution(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
    fluid_only=False,
    timing=False,
):
    # Create the grid

    t0 = time.perf_counter()
    if fluid_only:
        x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a, b)
    else:
        x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a + t_w, b)

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
    sloan_case.set_scaled_average_velocity(average_velocity)

    # Calculate scaled fields
    sloan_case.calculate_scaled_solution()

    t1 = time.perf_counter() - t0
    if timing: print(f"Me: time {t1:.2e}")
    return x, y, sloan_case


def compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
    timing=False,
):
    # This function can only compare the fluid domain as the HuntII solution
    # doesn't include solid domains

    x, y, yslanV, yslanH = yslan_solution(
        Ha,
        a,
        b,
        t_w,
        truncation,
        N_x,
        N_y,
        dyn_visc,
        conductivity_f,
        conductivity_w,
        permeability,
        average_velocity,
        timing=timing,
    )

    x, y, sloan_case = Sloan_solution(
        Ha,
        a,
        b,
        t_w,
        truncation,
        N_x,
        N_y,
        dyn_visc,
        conductivity_f,
        conductivity_w,
        permeability,
        average_velocity,
        fluid_only=True,
        timing=timing,
    )

    diff_uz = (
        sloan_case.scaled_velocity_z / np.mean(sloan_case.scaled_velocity_z)
        - yslanV/np.mean(yslanV)
    )

    rmse = mhdtools.statistics.rmse(diff_uz)
    print(f"Comparison - RMSE: {rmse}")

    return x, y, diff_uz


# Create a grid

a = 1  # height
b = 1  # width
N_x = 501  # num points in width
N_y = 501  # num points in height
t_w = 0.1  # Hartmann wall thickness

"""
Define a Hunt-II style flow with finite conductivity Hartmann walls
and perfectly insulating side walls, but considering the case of
thick walls. Hunt's solution assumes thin walls, but the Sloan 1966
solution includes the effect of Hartmann wall thickness.
"""

Ha = 1000  # Hartmann number
truncation = 170  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity_f = 1  # Fluid conductivity
conductivity_w = 0.1  # Solid conductivity
permeability = 1  # Fluid permeability
average_velocity = 1  # Average flow velocity

x, y, sloan_case = Sloan_solution(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
    timing=True,
)

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(
    x, y, sloan_case.scaled_velocity_z, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("sloan_comp_laminar_ducts_u_me.png")

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(
    x, y, sloan_case.scaled_B_field_z, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("sloan_comp_laminar_ducts_b_me.png")

print("\nMe:")
print("Pressure Gradient -dp/dz = %f Pa/m" % sloan_case.scaled_pressure_grad)
print("Flow Rate Q = %f [ND]" % sloan_case.Q)
print("Mean Velocity = %f [ND]" % sloan_case.average_velocity)
print("Rescaled Mean Velocity = %f m/s" % sloan_case.scaled_average_velocity)
print("Maximum Velocity = %f m/s" % np.max(sloan_case.scaled_velocity_z))


x, y, V, H = yslan_solution(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
    timing=True,
)

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(
    x, y, V, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("sloan_comp_laminar_ducts_u_yslan.png")

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(x, y, H, cmap="coolwarm", shading="gouraud")
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("sloan_comp_laminar_ducts_b_yslan.png")
print("\nYslan:")
print("No data")
# print("Pressure Gradient -dp/dz = %f Pa/m" % hunt_case.scaled_pressure_drop)
# print("Flow Rate Q = %f [ND]" % hunt_case.Q)
# print("Mean Velocity = %f [ND]" % hunt_case.average_velocity)
# print("Rescaled Mean Velocity = %f m/s" % hunt_case.scaled_average_velocity)
# print("Maximum Velocity = %f m/s" % np.max(hunt_case.scaled_velocity_z))

print(f"Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}")

x, y, uz_diff = compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
)

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(x, y, uz_diff, cmap="coolwarm", shading="gouraud")
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("sloan_comp_laminar_ducts_u_diff.png")


Ha = 100
t_w = 0.1
conductivity_w = 100
truncation = 100
print(
    f"Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
)

compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
)


Ha = 100
t_w = 1
conductivity_w = 1
truncation = 100
print(
    f"Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
)

compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
)


Ha = 100
t_w = 0.1
conductivity_w = 0
truncation = 100
print(
    f"Shercliff: Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
)

compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
)


Ha = 100
t_w = 0.1
conductivity_w = np.inf
truncation = 100
print(
    f"Hunt (perfect): Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
)

compare_solutions(
    Ha,
    a,
    b,
    t_w,
    truncation,
    N_x,
    N_y,
    dyn_visc,
    conductivity_f,
    conductivity_w,
    permeability,
    average_velocity,
)
