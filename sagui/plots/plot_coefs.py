import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import safety_gymnasium
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from safety_gymnasium.bases.base_task import BaseTask
import pandas as pd
import seaborn as sns
from random import random

from omnisafe.envs.sagui_envs import register_sagui_envs


# Extension
ext = 'pdf'

# Modify the physics constants of the environment


def modify_dyn(task: BaseTask, coef_dic: dict):
    model = task.model

    for name, mult in coef_dic.items():
        atr: np.ndarray = getattr(model, name)
        atr[:] *= mult


# Nice to have
def print_attr(obj):
    attr = [x for x in dir(obj) if x[0] != '_']
    print('\n'.join(attr))


# Mass and friction ranges
mass_range = [0.5, 1.0, 1.5]
fric_range = [0.5, 1.0, 1.5]

# Make the environment
register_sagui_envs()
env = safety_gymnasium.make('SafetyPointGuide0-v0')

# Matrix to store positions
pos_matrix = {}

# Plot different dynamics coefficients
for i, mass in enumerate(mass_range):  # enumerate(np.linspace(0.0001, 0.03, size)):
    for j, fric in enumerate(fric_range):  # enumerate(np.linspace(0, 0.008, size)):
        print(f'Mass: {mass}; Fric: {fric}')

        # Reset env and modify the dynamics
        env.reset(seed=0)
        terminated, truncated = False, False
        task: BaseTask = env.unwrapped.task

        modify_dyn(task, {'body_mass': mass, 'dof_damping': fric})

        # Run the trajectory
        positions = [task.agent.pos]
        while not terminated and not truncated:
            act = [1, 1]
            _, _, _, terminated, truncated, _ = env.step(act)
            positions.append(task.agent.pos)

        # Store positions in pos_matrix
        pos_matrix[(i, j)] = positions

# # Plot
# fig, axs = plt.subplots(len(fric_range), len(mass_range), figsize=(10, 10))
# for (i, j), positions in pos_matrix.items():
#     # Unpack the positions
#     positions = np.array(positions)
#     x_positions = positions[:, 0]
#     y_positions = positions[:, 1]

#     # Get mass and friction
#     mass = mass_range[i]
#     fric = fric_range[j]

#     # Subplot
#     ax: Axes = axs[i, j]
#     ax.set_title(f'Mass={"{:.1f}".format(mass)}; Damp={"{:.1f}".format(fric)}', fontsize=17)
#     ax.set_xlim(0.25, 1.75)
#     ax.set_ylim(-0.5, 1.0)

#     ax.plot(x_positions, y_positions)


# plt.tight_layout()
# plt.savefig('./plot.png')

data = []
for (i, j), positions in pos_matrix.items():
    positions = np.array(positions)
    mass = mass_range[i]
    fric = fric_range[j]

    for pos in positions:
        data.append({'x': pos[0], 'y': pos[1], 'mass': mass, 'friction': fric})

# Create a DataFrame
df = pd.DataFrame(data)

# Ensure 'x' and 'y' columns are numpy arrays (though they are by default in pandas)
df['x'] = df['x'].to_numpy()
df['y'] = df['y'].to_numpy()

# Initialize a FacetGrid with seaborn
g = sns.FacetGrid(df, row='friction', col='mass', margin_titles=True, height=4, aspect=1)

# Map the lineplot onto the grid
g.map(plt.plot, 'x', 'y')

# Set the axis limits
g.set(xlim=(0.25, 1.75), ylim=(-0.5, 1.0))

# Set titles for each subplot using row_var and col_var instead of 'friction' and 'mass'
g.set_titles(row_template="Friction ×{row_name}", col_template="Mass ×{col_name}", size=24)

# Add axis labels
g.set_axis_labels("X Position", "Y Position", size=16)

# Tick font size
g.set_xticklabels(labels=[], fontsize=16)
g.set_yticklabels(labels=[], fontsize=16)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'./plot.{ext}')
