import os
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from utils import read_final_simulation


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


path_to_data = '/Users/kostyansa/openfoam/data'
path_to_low = os.path.join(path_to_data, 'low_dim')
path_to_high = os.path.join(path_to_data, 'high_dim')
simulation = 'vel1'

simulation_low = os.path.join(path_to_low, simulation)
C_low, U_low, p_low = read_final_simulation(simulation_low)

simulation_high = os.path.join(path_to_high, simulation)
C_high, U_high, p_high = read_final_simulation(simulation_high)

interpol_p = CloughTocher2DInterpolator(C_low[:, :2], p_low)

predicted_p = np.nan_to_num(interpol_p(C_high[:, 0], C_high[:, 1]))

interpol_u_x = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 1])
interpol_u_y = CloughTocher2DInterpolator(C_low[:, :2], U_low[:, 2])

predicted_u_x = np.nan_to_num(interpol_u_x(C_high[:, 0], C_high[:, 1]))
predicted_u_y = np.nan_to_num(interpol_u_y(C_high[:, 0], C_high[:, 1]))

print(predicted_p)

fig, axes = plt.subplots(2, 2, squeeze=False)
original_ax = axes[0, 0]
triangulation = tri.Triangulation(C_high[:, 0], C_high[:, 1])

original_ax.tricontour(triangulation, p_high, levels=14, linewidths=0.5, colors='k')

predicted_ax = axes[0, 1]
predicted_ax.tricontour(triangulation, predicted_p, levels=14, linewidths=0.5, colors='k')

original_ax_u = axes[1, 0]
original_ax_u.quiver(C_high[:, 0], C_high[:, 1], U_high[:, 0], U_high[:, 1])

# Add labels and title
original_ax_u.set_xlabel("X")
original_ax_u.set_ylabel("Y")
original_ax_u.set_title("Original Velocity Vector Field (U)")

predicted_ax_u = axes[1, 1]
predicted_ax_u.quiver(C_high[:, 0], C_high[:, 1], predicted_u_x, predicted_u_y)

# Add labels and title
predicted_ax_u.set_xlabel("X")
predicted_ax_u.set_ylabel("Y")
predicted_ax_u.set_title("Predicted Velocity Vector Field (U)")

plt.savefig("pressure.png")
print(mape(p_high, predicted_p))

print(mape(U_high[:, :2], np.dstack((predicted_u_x, predicted_u_y))))


