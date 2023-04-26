import os
import matplotlib.pyplot as plt

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
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(3, activation='linear')
])

initial_learning_rate = 0.001
decay_rate = 0.9
decay_steps = 10000

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name="Adam"
)

model.compile(optimizer='adam', loss='mape')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="perceptron",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history = model.fit(df, target, validation_split=0.33, epochs=200, batch_size=512, callbacks=[model_checkpoint_callback])

simulation = 'vel1'

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

predicted = model.predict(data)[0]

# Plot the training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

fig, axes = plt.subplots(1, 2)
original = axes[0]
original.quiver(C_high[:, 0], C_high[:, 1], U_high[:, 0], U_high[:, 1])

# Add labels and title
original.set_xlabel("X")
original.set_ylabel("Y")
original.set_title("Velocity Vector Field (U)")

predicted_plot = axes[1]
predicted_plot.quiver(C_high[:, 0], C_high[:, 1], predicted[:, 0], predicted[:, 1])

# Add labels and title
predicted_plot.set_xlabel("X")
predicted_plot.set_ylabel("Y")
predicted_plot.set_title("Velocity Vector Field (U)")

# Display the plot
plt.savefig(f"high_field.png")
