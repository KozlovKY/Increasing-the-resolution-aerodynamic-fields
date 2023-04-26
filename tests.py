import os

import numpy as np
import openfoamparser_mai as Ofpp
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils import read_final_simulation

path_to_data = '/Users/kostyansa/openfoam/data'
number_of_neighbours = 6
columns = []

for i in range(number_of_neighbours):
    columns.append(f"r_{i}")
    columns.append(f"v_x_{i}")
    columns.append(f"v_y_{i}")
    columns.append(f"v_z_{i}")

data = []
target = []
path_to_low = os.path.join(path_to_data, 'low_dim')
path_to_high = os.path.join(path_to_data, 'high_dim')
for simulation in os.listdir(path_to_low):

    simulation_low = os.path.join(path_to_low, simulation)

    C_low, U_low, p_low = read_final_simulation(simulation_low)

    k_n = NearestNeighbors(n_neighbors=6)
    k_n.fit(C_low)

    simulation_high = os.path.join(path_to_high, simulation)
    C_high, U_high, p_high = read_final_simulation(simulation_high)

    rad_data, index_data = k_n.kneighbors(C_high, number_of_neighbours)
    for U, radi, indexes in zip(U_high, rad_data, index_data):
        row = []
        for radius, index in zip(radi, indexes):
            row.append(radius)
            U_n = U_low[index]
            for i in U_n:
                row.append(i)

        data.append(row)
        target.append(U)

df = pd.DataFrame(np.array(data), columns=columns)
target = np.array(target)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(96, activation='tanh', input_shape=(len(columns),)),
    tf.keras.layers.Dense(96, activation='tanh'),
    tf.keras.layers.Dense(96, activation='tanh'),
    tf.keras.layers.Dense(3, activation='tanh')
])

model.compile(optimizer='adam', loss='mse')

model.fit(df, target, validation_split=0.33, epochs=10000, batch_size=512)

