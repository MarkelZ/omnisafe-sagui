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
"""Implementation of Evaluator."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video
from safety_gymnasium.bases.base_task import BaseTask

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner import (
    ARCPlanner,
    CAPPlanner,
    CCEPlanner,
    CEMPlanner,
    RCEPlanner,
    SafeARCPlanner,
)
from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, make
from omnisafe.envs.wrapper import ActionRepeat, ActionScale, ObsNormalize, TimeLimit
from omnisafe.models.actor import ActorBuilder
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.actor_critic import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config


def _modify_dyn(task: BaseTask, coef_dict: dict):
    model = task.model
    for name, mult in coef_dict.items():
        atr: np.ndarray = getattr(model, name)
        atr[:] *= mult


class EvaluatorRobust:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms.

    Args:
        env (CMDP or None, optional): The environment. Defaults to None.
        actor (Actor or None, optional): The actor. Defaults to None.
        render_mode (str, optional): The render mode. Defaults to 'rgb_array'.
    """

    _cfgs: Config
    _save_dir: str
    _model_name: str
    _cost_count: torch.Tensor

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env: CMDP | None = None,
        actor: Actor | None = None,
        actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = None,
        dynamics: EnsembleDynamicsModel | None = None,
        planner: CEMPlanner
        | ARCPlanner
        | SafeARCPlanner
        | CCEPlanner
        | CAPPlanner
        | RCEPlanner
        | None = None,
        render_mode: str = 'rgb_array',
    ) -> None:
        """Initialize an instance of :class:`Evaluator`."""
        self._guide_env: CMDP | None = env
        self._actor: Actor | None = actor
        self._actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = actor_critic
        self._dynamics: EnsembleDynamicsModel | None = dynamics
        self._planner = planner
        self._dividing_line: str = '\n' + '#' * 50 + '\n'

        self._safety_budget: torch.Tensor
        self._safety_obs = torch.ones(1)
        self._cost_count = torch.zeros(1)
        self.__set_render_mode(render_mode)

    def __set_render_mode(self, render_mode: str) -> None:
        """Set the render mode.

        Args:
            render_mode (str, optional): The render mode. Defaults to 'rgb_array'.

        Raises:
            NotImplementedError: If the render mode is not implemented.
        """
        # set the render mode
        if render_mode in ['human', 'rgb_array', 'rgb_array_list']:
            self._render_mode: str = render_mode
        else:
            raise NotImplementedError('The render mode is not implemented.')

    def __load_cfgs(self, save_dir: str) -> None:
        """Load the config from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        self._cfgs = Config.dict2config(kwargs)

    # pylint: disable-next=too-many-branches
    def __load_model_and_env(
        self,
        save_dir: str,
        model_name: str,
        env_kwargs: dict[str, Any],
    ) -> None:
        """Load the model from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.
            model_name (str): Name of the model.
            env_kwargs (dict[str, Any]): Keyword arguments for the environment.

        Raises:
            FileNotFoundError: If the model is not found.
        """
        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        # Load the guide's environment
        self._guide_env = make(**env_kwargs)

        # Load the student's environment
        env_id_dict = {'SafetyPointGuide0-v0': 'SafetyPointStudent0-v0',
                       'SafetyPointGuide1-v0': 'SafetyPointStudent1-v0',
                       'SafetyPointGuide2-v0': 'SafetyPointStudent2-v0', }
        env_id = env_kwargs['env_id']

        assert env_id in env_id_dict, f'Unknown environment ID {env_id}.\nAllowed environments: {env_id_dict.keys()}'

        env_kwargs['env_id'] = env_id_dict[env_id]
        self._env = make(**env_kwargs)

        observation_space = self._guide_env.observation_space
        action_space = self._guide_env.action_space

        # if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
        #     self._safety_budget = (
        #         self._cfgs.algo_cfgs.safety_budget
        #         * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
        #         / (1 - self._cfgs.algo_cfgs.saute_gamma)
        #         / self._cfgs.algo_cfgs.max_ep_len
        #         * torch.ones(1)
        #     )
        assert isinstance(observation_space, Box), 'The observation space must be Box.'
        assert isinstance(action_space, Box), 'The action space must be Box.'

        if self._cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=observation_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
            self._guide_env = ObsNormalize(self._guide_env, device=torch.device('cpu'), norm=obs_normalizer)
        if self._guide_env.need_time_limit_wrapper:
            self._guide_env = TimeLimit(self._guide_env, device=torch.device('cpu'), time_limit=1000)
        self._guide_env = ActionScale(self._guide_env, device=torch.device('cpu'), low=-1.0, high=1.0)

        if hasattr(self._cfgs['algo_cfgs'], 'action_repeat'):
            self._guide_env = ActionRepeat(
                self._guide_env,
                device=torch.device('cpu'),
                times=self._cfgs['algo_cfgs']['action_repeat'],
            )
        if hasattr(self._cfgs, 'algo') and self._cfgs['algo'] in [
            'LOOP',
            'SafeLOOP',
            'PETS',
            'CAPPETS',
            'RCEPETS',
            'CCEPETS',
        ]:
            dynamics_state_space = (
                self._guide_env.coordinate_observation_space
                if self._guide_env.coordinate_observation_space is not None
                else self._guide_env.observation_space
            )
            assert self._guide_env.action_space is not None and isinstance(
                self._guide_env.action_space.shape,
                tuple,
            )
            if isinstance(self._guide_env.action_space, Box):
                action_space = self._guide_env.action_space
            else:
                raise NotImplementedError
            if self._cfgs['algo'] in ['LOOP', 'SafeLOOP']:
                self._actor_critic = ConstraintActorQCritic(
                    obs_space=dynamics_state_space,
                    act_space=action_space,
                    model_cfgs=self._cfgs.model_cfgs,
                    epochs=1,
                )
            if self._actor_critic is not None:
                self._actor_critic.load_state_dict(model_params['actor_critic'])
                self._actor_critic.to('cpu')
            self._dynamics = EnsembleDynamicsModel(
                model_cfgs=self._cfgs.dynamics_cfgs,
                device=torch.device('cpu'),
                state_shape=dynamics_state_space.shape,
                action_shape=action_space.shape,
                actor_critic=self._actor_critic,
                rew_func=None,
                cost_func=self._guide_env.get_cost_from_obs_tensor,
                terminal_func=None,
            )
            self._dynamics.ensemble_model.load_state_dict(model_params['dynamics'])
            self._dynamics.ensemble_model.to('cpu')
            if self._cfgs['algo'] in ['CCEPETS', 'RCEPETS', 'SafeLOOP']:
                algo_to_planner = {
                    'CCEPETS': (
                        'CCEPlanner',
                        {'cost_limit': self._cfgs['algo_cfgs']['cost_limit']},
                    ),
                    'RCEPETS': (
                        'RCEPlanner',
                        {'cost_limit': self._cfgs['algo_cfgs']['cost_limit']},
                    ),
                    'SafeLOOP': (
                        'SafeARCPlanner',
                        {
                            'cost_limit': self._cfgs['algo_cfgs']['cost_limit'],
                            'actor_critic': self._actor_critic,
                        },
                    ),
                }
            elif self._cfgs['algo'] in ['PETS', 'LOOP']:
                algo_to_planner = {
                    'PETS': ('CEMPlanner', {}),
                    'LOOP': ('ARCPlanner', {'actor_critic': self._actor_critic}),
                }
            elif self._cfgs['algo'] in ['CAPPETS']:
                lagrange: torch.nn.Parameter = torch.nn.Parameter(
                    model_params['lagrangian_multiplier'].to('cpu'),
                    requires_grad=False,
                )
                algo_to_planner = {
                    'CAPPETS': (
                        'CAPPlanner',
                        {
                            'cost_limit': self._cfgs['lagrange_cfgs']['cost_limit'],
                            'lagrange': lagrange,
                        },
                    ),
                }
            planner_name = algo_to_planner[self._cfgs['algo']][0]
            planner_special_cfgs = algo_to_planner[self._cfgs['algo']][1]
            planner_cls = globals()[f'{planner_name}']
            self._planner = planner_cls(
                dynamics=self._dynamics,
                planner_cfgs=self._cfgs.planner_cfgs,
                gamma=float(self._cfgs.algo_cfgs.gamma),
                cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
                dynamics_state_shape=dynamics_state_space.shape,
                action_shape=action_space.shape,
                action_max=1.0,
                action_min=-1.0,
                device='cpu',
                **planner_special_cfgs,
            )

        else:
            if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                observation_space = Box(
                    low=np.hstack((observation_space.low, -np.inf)),
                    high=np.hstack((observation_space.high, np.inf)),
                    shape=(observation_space.shape[0] + 1,),
                )
            actor_type = self._cfgs['model_cfgs']['actor_type']
            pi_cfg = self._cfgs['model_cfgs']['actor']
            weight_initialization_mode = self._cfgs['model_cfgs']['weight_initialization_mode']
            actor_builder = ActorBuilder(
                obs_space=observation_space,
                act_space=action_space,
                hidden_sizes=pi_cfg['hidden_sizes'],
                activation=pi_cfg['activation'],
                weight_initialization_mode=weight_initialization_mode,
            )
            self._actor = actor_builder.build_actor(actor_type)
            self._actor.load_state_dict(model_params['pi'])

            # This is for building the student!
            self.student_builder = ActorBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=[256, 256],
                activation='relu',
                weight_initialization_mode='kaiming_uniform',
            )
            self.student_untrained: GaussianSACActor = None

    # pylint: disable-next=too-many-locals
    def load_saved(
        self,
        save_dir: str,
        model_name: str,
        render_mode: str = 'rgb_array',
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 256,
        height: int = 256,
    ) -> None:
        """Load a saved model.

        Args:
            save_dir (str): The directory where the model is saved.
            model_name (str): The name of the model.
            render_mode (str, optional): The render mode, ranging from 'human', 'rgb_array',
                'rgb_array_list'. Defaults to 'rgb_array'.
            camera_name (str or None, optional): The name of the camera. Defaults to None.
            camera_id (int or None, optional): The id of the camera. Defaults to None.
            width (int, optional): The width of the image. Defaults to 256.
            height (int, optional): The height of the image. Defaults to 256.
        """
        # load the config
        self._save_dir = save_dir
        self._model_name = model_name

        self.__load_cfgs(save_dir)

        self.__set_render_mode(render_mode)

        env_kwargs = {
            'env_id': self._cfgs['env_id'],
            'num_envs': 1,
            'render_mode': self._render_mode,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }

        self.__load_model_and_env(save_dir, model_name, env_kwargs)

    def _obs_student_to_guide(self, current_obs: torch.Tensor):

        base_guide_env = self._guide_env.get_base_env()
        base_student_env = self._env.get_base_env()

        guide_size = self._guide_env.observation_space.shape[0]

        student_obs_dict = base_student_env.obs_space_dict
        guide_obs_dict = base_guide_env.obs_space_dict

        guide_keys = [k for k in guide_obs_dict.keys()]

        student_offset = 0
        teacher_offset = 0

        guide_obs_flat = np.zeros(guide_size, dtype=np.float32)
        for k in student_obs_dict.keys():
            obs = student_obs_dict[k]
            k_size = np.prod(obs.shape)
            if k in guide_keys:
                vals = current_obs[student_offset:student_offset + k_size]
                guide_obs_flat[teacher_offset:teacher_offset + k_size] = vals
                teacher_offset += k_size
            student_offset += k_size
        return torch.tensor(guide_obs_flat, dtype=torch.float32)

    # def _get_student_act(self, obs, deterministic):
    #     mean_1 = 0.25
    #     mean_2 = 0.0
    #     std_1 = 1.0
    #     std_2 = 1.0

    #     # act_1 is throttle and act_2 is steer
    #     act_1 = np.random.randn(1)[0] * std_1 + mean_1
    #     act_2 = np.random.randn(1)[0] * std_2 + mean_2
    #     act = torch.as_tensor([act_1, act_2], dtype=torch.float32)

    #     torch.clamp(act, -1.0, 1.0)

    #     return act

    def _rebuild_student(self):
        if self.student_untrained != None:
            self.student_untrained.net = None  # Pls garbage collect this
        self.student_untrained: GaussianSACActor = self.student_builder.build_actor('gaussian_sac')
        # self.student_untrained: MLPActor = self.student_builder.build_actor('mlp')
        # self.student_untrained._noise = 0.75

    def _get_student_act(self, obs, deterministic):
        return self.student_untrained.predict(obs, deterministic)

    def evaluate(
        self,
        coef_list: list[dict[str, float]],
        num_episodes: int = 1,
        cost_criteria: float = 1.0,
        deterministic: bool = False,
        process_name: str = None
    ) -> list[tuple[dict[str, float], float]]:
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): The cost criteria. Defaults to 1.0.

        Returns:
            (episode_rewards, episode_costs): The episode rewards and costs.

        Raises:
            ValueError: If the environment and the policy are not provided or created.
        """

        if self._guide_env is None or (self._actor is None and self._planner is None):
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.',
            )

        torch.set_num_threads(1)  # A single thread is enough for feed forward
        result = []
        for i, coef_dict in enumerate(coef_list):
            # Print progress
            if process_name != None:
                progress = 100. * i / len(coef_list)
                print(f'[{process_name}]: Progress {progress:.1f}%')

            costs = []
            for ep in range(num_episodes):
                # Rebuild the student with new weights
                self._rebuild_student()

                # Reset env
                obs, _ = self._env.reset()
                self._safety_obs = torch.ones(1)
                ep_ret, ep_cost, length = 0.0, 0.0, 0.0

                # Modify dynamics
                base_env = self._env.get_base_env()
                task: BaseTask = base_env.unwrapped.task
                _modify_dyn(task, coef_dict)

                recover = False
                done = False
                while not done:
                    with torch.no_grad():
                        if recover:
                            obs = self._obs_student_to_guide(obs)
                            act = self._actor.predict(obs, deterministic=deterministic)
                        else:
                            act = self._get_student_act(obs, deterministic)

                    obs, rew, cost, terminated, truncated, _ = self._env.step(act)

                    ep_ret += rew.item()
                    ep_cost += (cost_criteria**length) * cost.item()

                    recover = cost.item() > 0.

                    length += 1

                    done = bool(terminated or truncated)

                costs.append(ep_cost)

            avg_cost = np.mean(a=costs)
            v = (coef_dict, avg_cost)
            result.append(v)

        print(f'[{process_name}]: Done!')

        return result

    def evaluate_student(
        self,
        coef_list: list[dict[str, float]],
        num_episodes: int = 1,
        cost_criteria: float = 1.0,
        deterministic: bool = False,
        process_name: str = None
    ) -> list[tuple[dict[str, float], float]]:
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): The cost criteria. Defaults to 1.0.

        Returns:
            (episode_rewards, episode_costs): The episode rewards and costs.

        Raises:
            ValueError: If the environment and the policy are not provided or created.
        """

        if self._guide_env is None or (self._actor is None and self._planner is None):
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.',
            )

        torch.set_num_threads(1)  # A single thread is enough for feed forward
        result = []
        for i, coef_dict in enumerate(coef_list):
            # Print progress
            if process_name != None:
                progress = 100. * i / len(coef_list)
                print(f'[{process_name}]: Progress {progress:.1f}%')

            costs = []
            for ep in range(num_episodes):
                obs, _ = self._env.reset()
                self._safety_obs = torch.ones(1)
                ep_ret, ep_cost, length = 0.0, 0.0, 0.0

                # Modify dynamics
                base_env = self._env.get_base_env()
                task: BaseTask = base_env.unwrapped.task
                _modify_dyn(task, coef_dict)

                done = False
                while not done:
                    with torch.no_grad():
                        act = self._get_student_act(obs, deterministic)

                    obs, rew, cost, terminated, truncated, _ = self._env.step(act)

                    ep_ret += rew.item()
                    ep_cost += (cost_criteria**length) * cost.item()

                    length += 1

                    done = bool(terminated or truncated)

                costs.append(ep_cost)

            avg_cost = np.mean(a=costs)
            v = (coef_dict, avg_cost)
            result.append(v)

        print(f'[{process_name}]: Done!')

        return result

    @property
    def fps(self) -> int:
        """The fps of the environment.

        Raises:
            AssertionError: If the environment is not provided or created.
            AtrributeError: If the fps is not found.
        """
        assert (
            self._guide_env is not None
        ), 'The environment must be provided or created before getting the fps.'
        try:
            fps = self._guide_env.metadata['render_fps']
        except AttributeError:
            fps = 30
            warnings.warn('The fps is not found, use 30 as default.', stacklevel=2)

        return fps

    def render(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
        self,
        coef_dict: dict[str, float],
        num_episodes: int = 1,
        save_replay_path: str | None = None,
        max_render_steps: int = 2000,
        cost_criteria: float = 1.0,
    ) -> None:  # pragma: no cover
        """Render the environment for one episode.

        Args:
            num_episodes (int, optional): The number of episodes to render. Defaults to 1.
            save_replay_path (str or None, optional): The path to save the replay video. Defaults to
                None.
            max_render_steps (int, optional): The maximum number of steps to render. Defaults to 2000.
            cost_criteria (float, optional): The discount factor for the cost. Defaults to 1.0.
        """
        assert (
            self._env is not None
        ), 'The environment must be provided or created before rendering.'
        assert (
            self._actor is not None or self._planner is not None
        ), 'The policy or planner must be provided or created before rendering.'
        if save_replay_path is None:
            save_replay_path = os.path.join(self._save_dir, 'video', self._model_name.split('.')[0])
        result_path = os.path.join(save_replay_path, 'result.txt')
        print(self._dividing_line)
        print(f'Saving the replay video to {save_replay_path},\n and the result to {result_path}.')
        print(self._dividing_line)

        horizon = 1000
        frames = []
        obs, _ = self._env.reset()
        if self._render_mode == 'human':
            self._env.render()
        elif self._render_mode == 'rgb_array':
            frames.append(self._env.render())

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []

        for episode_idx in range(num_episodes):
            # Modify dynamics
            base_env = self._env.get_base_env()
            task: BaseTask = base_env.unwrapped.task
            _modify_dyn(task, coef_dict)

            self._rebuild_student()
            print("Rendering the STUDENT!!")

            self._safety_obs = torch.ones(1)
            step = 0
            done = False
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0
            while (
                not done and step <= max_render_steps
            ):  # a big number to make sure the episode will end
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    obs = torch.cat([obs, self._safety_obs], dim=-1)

                with torch.no_grad():
                    # obs = self._obs_student_to_guide(obs)
                    # act = self._actor.predict(obs, deterministic=True)
                    act = self._get_student_act(obs, False)

                obs, rew, cost, terminated, truncated, _ = self._env.step(act)
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
                    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma
                step += 1
                done = bool(terminated or truncated)
                ep_ret += rew.item()
                ep_cost += (cost_criteria**length) * cost.item()
                if (
                    'EarlyTerminated' in self._cfgs['algo']
                    and ep_cost >= self._cfgs.algo_cfgs.cost_limit
                ):
                    terminated = torch.as_tensor(True)
                length += 1

                if self._render_mode == 'rgb_array':
                    frames.append(self._env.render())

            if self._render_mode == 'rgb_array_list':
                frames = self._env.render()
            if save_replay_path is not None:
                save_video(
                    frames,
                    save_replay_path,
                    fps=self.fps,
                    episode_trigger=lambda x: True,
                    video_length=horizon,
                    episode_index=episode_idx,
                    name_prefix='eval',
                )
            self._env.reset()
            frames = []
            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)
            with open(result_path, 'a+', encoding='utf-8') as f:
                print(f'Episode {episode_idx+1} results:', file=f)
                print(f'Episode reward: {ep_ret}', file=f)
                print(f'Episode cost: {ep_cost}', file=f)
                print(f'Episode length: {length}', file=f)
        with open(result_path, 'a+', encoding='utf-8') as f:
            print(self._dividing_line)
            print('Evaluation results:', file=f)
            print(f'Average episode reward: {np.mean(episode_rewards)}', file=f)
            print(f'Average episode cost: {np.mean(episode_costs)}', file=f)
            print(f'Average episode length: {np.mean(episode_lengths)}', file=f)
