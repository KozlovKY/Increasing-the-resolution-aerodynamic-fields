import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils import read_final_simulation

import os
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
import numpy as np

path_to_data = '/Users/kostyansa/openfoam/data'
number_of_neighbours = 6
columns = ['p', 'u_x', 'u_y']

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

    k_n = NearestNeighbors(n_neighbors=number_of_neighbours)
    k_n.fit(C_low)

    simulation_high = os.path.join(path_to_high, simulation)
    C_high, U_high, p_high = read_final_simulation(simulation_high)

    interpol_p = CloughTocher2DInterpolator(C_low[:, :2], p_low)

    predicted_p = np.nan_to_num(interpol_p(C_high[:, 0], C_high[:, 1]))

    interpol_u_x = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 1])
    interpol_u_y = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 2])

    predicted_u_x = np.nan_to_num(interpol_u_x(C_high[:, 0], C_high[:, 1]))
    predicted_u_y = np.nan_to_num(interpol_u_y(C_high[:, 0], C_high[:, 1]))

    rad_data, index_data = k_n.kneighbors(C_high, number_of_neighbours)
    for U, radi, indexes, p, u_x, u_y in zip(U_high, rad_data, index_data, predicted_p, predicted_u_x, predicted_u_y):
        row = [p, u_x, u_y]
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
    tf.keras.layers.Dense(96, activation='linear', input_shape=(len(columns),)),
    tf.keras.layers.Dense(96, activation='linear'),
    tf.keras.layers.Dense(96, activation='linear'),
    tf.keras.layers.Dense(32, activation='linear'),
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
    filepath="model/perceptron.h5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history = model.fit(df, target, validation_split=0.33, epochs=200, batch_size=128, callbacks=[model_checkpoint_callback])

simulation = 'vel1'

simulation_low = os.path.join(path_to_low, simulation)
simulation_high = os.path.join(path_to_high, simulation)


C_low, U_low, p_low = read_final_simulation(simulation_low)

k_n = NearestNeighbors(n_neighbors=number_of_neighbours)
k_n.fit(C_low)

C_high, U_high, p_high = read_final_simulation(simulation_high)

interpol_p = CloughTocher2DInterpolator(C_low[:, :2], p_low)

predicted_p = np.nan_to_num(interpol_p(C_high[:, 0], C_high[:, 1]))

interpol_u_x = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 1])
interpol_u_y = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 2])

predicted_u_x = np.nan_to_num(interpol_u_x(C_high[:, 0], C_high[:, 1]))
predicted_u_y = np.nan_to_num(interpol_u_y(C_high[:, 0], C_high[:, 1]))

data = []

rad_data, index_data = k_n.kneighbors(C_high, number_of_neighbours)
for U, radi, indexes, p, u_x, u_y in zip(U_high, rad_data, index_data, predicted_p, predicted_u_x, predicted_u_y):
    row = [p, u_x, u_y]
    for radius, index in zip(radi, indexes):
        row.append(radius)
        U_n = U_low[index]
        for i in U_n:
            row.append(i)
    data.append(row)

model = tf.keras.models.load_model('model/perceptron.h5')

predicted = model.predict(np.array(data))

fig, axes = plt.subplots(2, 2)
original = axes[0, 0]
original.quiver(C_high[:, 0], C_high[:, 1], U_high[:, 0], U_high[:, 1])

# Add labels and title
original.set_xlabel("X")
original.set_ylabel("Y")
original.set_title("Velocity Vector Field (U)")

predicted_plot = axes[0, 1]
predicted_plot.quiver(C_high[:, 0], C_high[:, 1], predicted[:, 0], predicted[:, 1])

# Add labels and title
predicted_plot.set_xlabel("X")
predicted_plot.set_ylabel("Y")
predicted_plot.set_title("Velocity Vector Field (U)")

original_ax_u = axes[1, 0]
original_ax_u.scatter(C_high[:, 0], C_high[:, 1], c=np.sqrt(np.square(U_high[:, 0]) + np.square(U_high[:, 1]))/np.sqrt(np.square(np.max(U_high[:, 0])) + np.square(np.max(U_high[:, 1]))), marker='.')

# Add labels and title
original_ax_u.set_xlabel("X")
original_ax_u.set_ylabel("Y")
original_ax_u.set_title("Original Velocity Vector Field (U)")

predicted_ax_u = axes[1, 1]
predicted_ax_u.scatter(C_high[:, 0], C_high[:, 1], c=np.sqrt(np.square(predicted[:, 0]) + np.square(predicted[:, 1]))/np.sqrt(np.square(np.max(U_high[:, 0])) + np.square(np.max(U_high[:, 1]))), marker='.')

# Add labels and title
predicted_ax_u.set_xlabel("X")
predicted_ax_u.set_ylabel("Y")
predicted_ax_u.set_title("Predicted Velocity Vector Field (U)")

# Display the plot
plt.savefig(f"high_field.png")
