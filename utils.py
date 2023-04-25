import os
from typing import Union
import numpy as np

import openfoamparser_mai as Ofpp


PathLike = Union[bytes, str, os.PathLike]

LOW_DIM = 'low_dim'
HIGH_DIM = 'high_dim'


def max_timestep(simulation: PathLike) -> PathLike:
    timestep = str(max(map(lambda x: int(x) if x.isdigit() else -1, os.listdir(simulation))))
    return os.path.join(simulation, timestep)


def read_mesh_centers(timestep: PathLike):
    Cx = Ofpp.parse_internal_field(os.path.join(timestep, 'Cx'))
    Cy = Ofpp.parse_internal_field(os.path.join(timestep, 'Cy'))
    Cz = Ofpp.parse_internal_field(os.path.join(timestep, 'Cz'))
    return np.dstack((Cx, Cy, Cz))[0]


def read_speed_vector_field(timestep: PathLike):
    U = Ofpp.parse_internal_field(os.path.join(timestep, 'U'))
    return U


def read_pressure_field(timestep: PathLike):
    p = Ofpp.parse_internal_field(os.path.join(timestep, 'p'))
    return p


def read_timestep(timestep: PathLike):
    C = read_mesh_centers(timestep)
    U = read_speed_vector_field(timestep)
    p = read_pressure_field(timestep)
    return C, U, p


def read_final_simulation(simulation: PathLike):
    timestep = max_timestep(simulation)
    return read_timestep(timestep)


def read_geometry(path_to_geometry: PathLike):
    low_dim_path = os.path.join(path_to_geometry, LOW_DIM)
    high_dim_path = os.path.join(path_to_geometry, HIGH_DIM)

    low_dim = np.array(list(map(read_final_simulation, map(lambda x: os.path.join(path_to_geometry, x), os.listdir(low_dim_path)))))
    high_dim = np.array(list(map(read_final_simulation, map(lambda x: os.path.join(path_to_geometry, x), os.listdir(high_dim_path)))))

    return low_dim, high_dim


def read_timesteps(simulation: PathLike):
    timesteps = []
    for timestep in sorted(filter(lambda x: x.isdigit(), os.listdir(simulation))):
        timesteps.append((float(timestep), read_timestep(os.path.join(simulation, timestep))))
    return timesteps
