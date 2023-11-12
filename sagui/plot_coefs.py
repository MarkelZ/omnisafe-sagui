import numpy as np
import matplotlib.pyplot as plt
import safety_gymnasium
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from safety_gymnasium.bases.base_task import BaseTask


# Modify the physics constants of the environment
def modify_dyn(task: BaseTask, coef_dic: dict):
    model = task.model
    for coef, val in coef_dic.items():
        atr = getattr(model, coef)
        for index, value in np.ndenumerate(atr):
            atr[index] = value + val


# Nice to have
def print_attr(obj):
    attr = [x for x in dir(obj) if x[0] != '_']
    print('\n'.join(attr))


# Create a 4x4 grid of subplots
size = 4
fig, axs = plt.subplots(size, size, figsize=(10, 10))

# Make the environment
env = safety_gymnasium.make('SafetyPointGoal0-v0')

# Plot different dynamics coefficients
for i, mass in enumerate(np.linspace(0.0001, 0.03, size)):
    for j, fric in enumerate(np.linspace(0, 0.008, size)):
        print(f'Mass: {mass}; Fric: {fric}')

        # Reset env and modify the dynamics
        env.reset(seed=0)
        terminated, truncated = False, False
        task: BaseTask = env.unwrapped.task
        modify_dyn(task, {'body_mass': mass, 'dof_frictionloss': fric})

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
        ax = axs[i, j]
        ax.plot(x_positions, y_positions)
        ax.set_title(f'Mass={"{:.4f}".format(mass)}; Fric={"{:.4f}".format(fric)}')
        ax.set_xlim(-0.1, 0.9)
        ax.set_ylim(-0.1, 0.9)

plt.tight_layout()
plt.savefig('./plot.png')
