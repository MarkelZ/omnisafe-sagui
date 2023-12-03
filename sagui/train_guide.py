# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy from custom dict with OmniSafe."""

import omnisafe
from omnisafe.envs.sagui_envs import register_sagui_envs


if __name__ == '__main__':
    register_sagui_envs()
    env_id = 'SafetyPointGuide1-v0'

    cfgs = {
        'train_cfgs': {
            'total_steps': 1500000,
        },
        'algo_cfgs': {
            'alpha': 1,
            'cost_normalize': False,
            'gamma': 0.99,
            'steps_per_epoch': 30000,
            'update_iters': 100,
            'batch_size': 32,
            'size': 1000000,  # Replay buffer
            'start_learning_steps': 500,
            'warmup_epochs': 1,
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [32, 32],
                'lr': 1e-3,
            },
            'critic': {
                'hidden_sizes': [32, 32],
                'lr': 1e-3,
            }
        },
        'lagrange_cfgs': {
            'cost_limit': 5.0,
            'lagrangian_multiplier_init': 0.000,
            'lambda_lr': 50 * 1e-3,
            'lambda_optimizer': 'Adam',
        },
        'logger_cfgs': {
            'save_model_freq': 25,
        },
    }

    agent = omnisafe.Agent('SACLagB', env_id, custom_cfgs=cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
