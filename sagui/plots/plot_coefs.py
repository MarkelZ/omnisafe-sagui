import os
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

# Mass and friction ranges
mass_range = [0.5, 1.0, 1.5]
fric_range = [0.5, 1.0, 1.5]

# Path to save positions file
POSITIONS_PATH = './coef_positions.txt'


def generate_positions_file(path):
    # Modify the physics constants of the environment
    def modify_dyn(task: BaseTask, coef_dic: dict):
        model = task.model

        for name, mult in coef_dic.items():
            atr: np.ndarray = getattr(model, name)
            atr[:] *= mult

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

            # From numpy aray to list
            positions = [[x for x in pos] for pos in positions]

            # Store positions in pos_matrix
            pos_matrix[(i, j)] = positions

    # Write matrix to file
    with open(path, 'w') as file:
        file.write(str(pos_matrix))

    return pos_matrix


if not os.path.isfile(POSITIONS_PATH):
    # If the positions have not been generated yet, generate them
    pos_matrix = generate_positions_file(POSITIONS_PATH)
else:
    # Otherwise, load the positions from file
    with open(POSITIONS_PATH) as file:
        positions_src = ''.join(file.readlines())

    pos_matrix = eval(positions_src)

# Convert to array format for passing to pandas table
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
g.set_titles(row_template="Frict. ×{row_name}", col_template="Mass ×{col_name}", size=32)

# Add axis labels
g.set_axis_labels("X", "Y", size=24)

# Tick font size
g.set_xticklabels(labels=[], fontsize=16)
g.set_yticklabels(labels=[], fontsize=16)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'./plot.{ext}')
