import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import safety_gymnasium
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from safety_gymnasium.bases.base_task import BaseTask

from omnisafe.envs.sagui_envs import register_sagui_envs


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


# Create a 3x3 grid of subplots
size = 3
fig, axs = plt.subplots(size, size, figsize=(10, 10))

# Make the environment
register_sagui_envs()
# env = safety_gymnasium.make('SafetyPointGoal0-v0')
env = safety_gymnasium.make('SafetyPointGuide0-v0')

# Plot different dynamics coefficients
for i, mass in enumerate([0.5, 1.0, 1.5]):  # enumerate(np.linspace(0.0001, 0.03, size)):
    for j, fric in enumerate([0.5, 1.0, 1.5]):  # enumerate(np.linspace(0, 0.008, size)):
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

        # Unpack the positions
        positions = np.array(positions)
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        # Subplot
        ax: Axes = axs[i, j]
        ax.set_title(f'Mass={"{:.1f}".format(mass)}; Damp={"{:.1f}".format(fric)}', fontsize=17)
        ax.set_xlim(0.25, 1.75)
        ax.set_ylim(-0.5, 1.0)
        ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)

        ax.plot(x_positions, y_positions)


plt.tight_layout()
plt.savefig('./plot.png')
