from omnisafe.envs.sagui_envs import register_sagui_envs
import safety_gymnasium


register_sagui_envs()

env = safety_gymnasium.make('SafetyPointGuide0-v0', render_mode='human')
obs, info = env.reset()  # pylint: disable=unused-variable
# Use below to specify seed.
# obs, _ = env.reset(seed=0)
terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
while True:
    if terminated or truncated:
        print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
        ep_ret, ep_cost = 0, 0
        obs, info = env.reset()  # pylint: disable=unused-variable
    assert env.observation_space.contains(obs)
    act = env.action_space.sample()
    assert env.action_space.contains(act)
    # pylint: disable-next=unused-variable
    obs, reward, cost, terminated, truncated, info = env.step(act)

    ep_ret += reward
    ep_cost += cost
