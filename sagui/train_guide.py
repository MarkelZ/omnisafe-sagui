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
    env_id = 'SafetyPointGuide0-v0'

    cfgs = {
        'algo_cfgs': {
            'alpha': 0.00001,
            'cost_normalize': False
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [64, 64],
                'lr': 0.000005,
            },
            'critic': {
                'hidden_sizes': [64, 64],
                'lr': 0.001,
            }
        },
        'lagrange_cfgs': {
            'cost_limit': 5.0,
            'lagrangian_multiplier_init': 0.000,
            'lambda_lr': 0.0000005,
            'lambda_optimizer': 'Adam',
        }
    }

    agent = omnisafe.Agent('SACLagB', env_id, custom_cfgs=cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
