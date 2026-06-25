import mhdtools
import mhdtools.analytic
import numpy as np
import matplotlib.pyplot as plt
import time

# Define functions for Hunt and Sloan solutions


def Hunt_solution(
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

    t0 = time.perf_counter()

    # Create the grid

    # x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a + t_w, b)
    x, y = mhdtools.analytic.makeXYVectors(N_x, N_y, a, b)

    # calculate conductivity ratio
    dB = (t_w * conductivity_w) / (
        a * conductivity_f
    )  # Hartmann wall conduction ratio

    # Create the case
    hunt_case = mhdtools.analytic.HuntII(
        Ha,
        a,
        b,
        dB,
        truncation,
        x,
        y,
        dyn_visc,
        conductivity_f,
        permeability,
    )

    # Solve the case
    hunt_case.analytic_solve()

    # Constrain average velocity
    hunt_case.set_scaled_average_velocity(average_velocity)

    # Calculate scaled fields
    hunt_case.calculate_scaled_solution()

    t1 = time.perf_counter() - t0
    if timing:
        print(f"\nHunt time: {t1:.2e}s")

    return x, y, hunt_case


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

    t0 = time.perf_counter()

    # Create the grid

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
    if timing:
        print(f"\nSloan time: {t1:.2e}s")
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

    x, y, hunt_case = Hunt_solution(
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

    diff_uz = sloan_case.scaled_velocity_z - hunt_case.scaled_velocity_z

    rmse = mhdtools.statistics.rmse(diff_uz)
    print(f"Comparison - RMSE: {rmse}")

    return x, y, diff_uz


# Create a grid

a = 1  # height
b = 1  # width
N_x = 501  # num points in width
N_y = 501  # num points in height
t_w = 1  # Hartmann wall thickness

"""
Define a Hunt-II style flow with finite conductivity Hartmann walls
and perfectly insulating side walls, but considering the case of
thick walls. Hunt's solution assumes thin walls, but the Sloan 1966
solution includes the effect of Hartmann wall thickness.
"""

Ha = 500  # Hartmann number
truncation = 70  # Number of Fourier iterations
dyn_visc = 1  # Dynamic viscosity
conductivity_f = 1  # Fluid conductivity
conductivity_w = 1  # Solid conductivity
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
plt.savefig("ex_3a_laminar_ducts_u_Sloan.png")

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(
    x, y, sloan_case.scaled_B_field_z, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_3a_laminar_ducts_b_Sloan.png")

print("\nSloan:")
print("Pressure Gradient -dp/dz = %f Pa/m" % sloan_case.scaled_pressure_grad)
print("Flow Rate Q = %f [ND]" % sloan_case.Q)
print("Mean Velocity = %f [ND]" % sloan_case.average_velocity)
print("Rescaled Mean Velocity = %f m/s" % sloan_case.scaled_average_velocity)
print("Maximum Velocity = %f m/s" % np.max(sloan_case.scaled_velocity_z))


x, y, hunt_case = Hunt_solution(
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
    x, y, hunt_case.scaled_velocity_z, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_3a_laminar_ducts_u_HuntII.png")

fig = plt.figure(figsize=(10, 8))
ax1 = plt.pcolormesh(
    x, y, hunt_case.scaled_B_field_z, cmap="coolwarm", shading="gouraud"
)
plt.xlabel("x")
plt.ylabel("y")
cb = plt.colorbar()
fig.tight_layout()
plt.savefig("ex_3a_laminar_ducts_b_HuntII.png")

print("\nHunt:")
print("Pressure Gradient -dp/dz = %f Pa/m" % hunt_case.scaled_pressure_drop)
print("Flow Rate Q = %f [ND]" % hunt_case.Q)
print("Mean Velocity = %f [ND]" % hunt_case.average_velocity)
print("Rescaled Mean Velocity = %f m/s" % hunt_case.scaled_average_velocity)
print("Maximum Velocity = %f m/s" % np.max(hunt_case.scaled_velocity_z))

print(
    f"Ha = {Ha}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
)

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
plt.savefig("ex_3a_laminar_ducts_diff.png")

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

print("\nVarying Ha:")
Ha = [20, 100, 500, 1000, 1e4, 2e4]
t_w = 0.1
conductivity_w = 1
truncation = 100

for Ha_val in Ha:
    print(
        f"Ha = {Ha_val}, t_w = {t_w}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
    )

    compare_solutions(
        Ha_val,
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


print("\nVarying t_w:")
Ha = 500
t_w = [0.01, 0.1, 1, 2, 5]
conductivity_w = 1
truncation = 100

for tw_val in t_w:
    print(
        f"Ha = {Ha}, t_w = {tw_val}, sigma_w = {conductivity_w}, Fourier terms = {truncation}"
    )

    compare_solutions(
        Ha,
        a,
        b,
        tw_val,
        truncation,
        N_x,
        N_y,
        dyn_visc,
        conductivity_f,
        conductivity_w,
        permeability,
        average_velocity,
    )


print("\nVarying sigma_w:")
Ha = 500
t_w = 0.1
conductivity_w = [0.01, 0.1, 1, 2, 5, 10, 100, 1e3, 1e4]
truncation = 100

for sigmaw_val in conductivity_w:
    print(
        f"Ha = {Ha}, t_w = {t_w}, sigma_w = {sigmaw_val}, Fourier terms = {truncation}"
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
        sigmaw_val,
        permeability,
        average_velocity,
    )
