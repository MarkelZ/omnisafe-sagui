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


if __name__ == '__main__':
    env_id = 'SafetyPointGoal1-v0'

    custom_cfgs = {
        'algo_cfgs': {
            'start_learning_steps': 0
        },
        'train_cfgs': {
            'torch_threads': 4,
        },
        'logger_cfgs': {
            'save_model_freq': 25
        },
    }

    agent = omnisafe.Agent('DDPGAdvAct', env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
