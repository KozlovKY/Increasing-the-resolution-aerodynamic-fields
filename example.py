import os
from typing import Union
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import openfoamparser_mai as Ofpp

from scipy.interpolate import griddata

PathLike = Union[str, os.PathLike]


def max_timestep(simulation: PathLike) -> PathLike:
    timestep = str(max(map(lambda x: int(x) if x.isdigit() else -1, os.listdir(simulation))))
    return os.path.join(simulation, timestep)


def read_mesh_centers(simulation: PathLike):
    timestep = max_timestep(simulation)
    Cx = Ofpp.parse_internal_field(os.path.join(timestep, 'Cx'))
    Cy = Ofpp.parse_internal_field(os.path.join(timestep, 'Cy'))
    Cz = Ofpp.parse_internal_field(os.path.join(timestep, 'Cz'))
    return np.dstack((Cx, Cy, Cz))[0]


def read_speed_vector_field(simulation: PathLike):
    timestep = max_timestep(simulation)
    U = Ofpp.parse_internal_field(os.path.join(timestep, 'U'))
    return U


def read_pressure_field(simulation: PathLike):
    timestep = max_timestep(simulation)
    p = Ofpp.parse_internal_field(os.path.join(timestep, 'p'))
    return p


path_to_data = './data/hig_dim'
simulation_name = 'vel1'
saveOutput = 1


simulation = os.path.join(path_to_data, simulation_name)

C = read_mesh_centers(simulation)
U = read_speed_vector_field(simulation)
p = read_pressure_field(simulation)

fig, ax = plt.subplots()
ax.quiver(C[:, 0], C[:, 1], U[:, 0], U[:, 1])

# Add labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Velocity Vector Field (U)")

# Display the plot
plt.savefig(f"high_field.png")
