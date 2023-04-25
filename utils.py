import os
from typing import Union
import numpy as np

import openfoamparser_mai as Ofpp


PathLike = Union[str, os.PathLike]

LOW_DIM = 'low_dim'
HIGH_DIM = 'high_dim'


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


def read_simulation(simulation: PathLike):
    C = read_mesh_centers(simulation)
    U = read_speed_vector_field(simulation)
    p = read_pressure_field(simulation)
    return C, U, p


def read_geometry(path_to_geometry: PathLike):
    low_dim_path = os.path.join(path_to_geometry, LOW_DIM)
    high_dim_path = os.path.join(path_to_geometry, HIGH_DIM)

    low_dim = np.array(list(map(read_simulation, map(lambda x: os.path.join(path_to_geometry, x), os.listdir(low_dim_path)))))
    high_dim = np.array(list(map(read_simulation, map(lambda x: os.path.join(path_to_geometry, x), os.listdir(high_dim_path)))))

    return low_dim, high_dim