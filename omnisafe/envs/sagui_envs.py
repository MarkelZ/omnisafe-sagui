import copy
from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.registration import register


# Register sagui environments with safety_gymnasium
# https://github.com/PKU-Alignment/safety-gymnasium/blob/6c777c6f892d4db400dec4a4f30f24db0dd52fde/safety_gymnasium/__init__.py#L55
def register_sagui_envs() -> None:
    env_id = 'SafetyPointGuide0-v0'
    config = {'agent_name': 'Point'}
    kwargs = {'config': config, 'task_id': env_id}
    register(id=env_id, entry_point='omnisafe.envs.sagui_builder:SaguiBuilder',
             kwargs=kwargs, max_episode_steps=1000)


# Took it from
# https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/tasks
class GuideLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        # self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=8, keepout=0.22))
        self._add_geoms(Sigwalls(num=4, locate_factor=3.2, is_constrained=True))
        self._add_free_geoms(Vases(num=1, is_constrained=False, keepout=0.18))

        # self.last_robot_pos = None
        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 8
        self.vases.num = 8
        self.vases.is_constrained = True

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0

        # Distance bonus here

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # self.build_goal_position()
        # self.last_robot_pos = self.agent.pos()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return False  # self.dist_goal() <= self.goal.size
