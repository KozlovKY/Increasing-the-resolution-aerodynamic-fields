import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CloughTocher2DInterpolator
from pykrige import UniversalKriging

import numpy as np

from utils import read_mesh_centers, read_speed_vector_field, read_pressure_field, read_final_simulation

path_to_data = '/Users/kostyansa/openfoam/data/hig_dim'
simulation_name = 'vel2'
saveOutput = 1


simulation = os.path.join(path_to_data, simulation_name)

C_high, U_high, p_high = read_final_simulation(simulation)

vmin = np.nanmin(p_high)
vmax = np.nanmax(p_high)

fig, ax = plt.subplots()

contour = plt.tricontourf(C_high[:, 0], C_high[:, 1], p_high, vmin=vmin, vmax=vmax)
fig.colorbar(contour, ax=ax, orientation='vertical')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Velocity Vector Field (U)")

# Display the plot
plt.savefig(f"original.png")

path_to_data = '/Users/kostyansa/openfoam/data/low_dim'
simulation_name = 'vel1'


simulation = os.path.join(path_to_data, simulation_name)

C_low, U_low, p_low = read_final_simulation(simulation)

krieg = UniversalKriging(C_low[:, 0], C_low[:, 1], p_low, variogram_model='linear')

z, _ = krieg.execute(style='points', xpoints=C_high[:, 0], ypoints=C_high[:, 1])

fig, ax = plt.subplots()
contour = plt.tricontourf(C_high[:, 0], C_high[:, 1], z, vmin=vmin, vmax=vmax)
fig.colorbar(contour, ax=ax, orientation='vertical')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Velocity Vector Field (U)")

# Display the plot
plt.savefig(f"aprox.png")


