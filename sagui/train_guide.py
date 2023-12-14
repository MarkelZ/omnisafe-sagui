import omnisafe
from omnisafe.envs.sagui_envs import register_sagui_envs


if __name__ == '__main__':
    envs = ['SafetyPointGuide0-v0', 'SafetyPointGuide1-v0', 'SafetyPointGuide2-v0']
    algos = ['DDPGLagUnfold', 'DDPGRandAct']
    
    for alg in algos:
        for env_id in envs:
            register_sagui_envs()

            cfgs = {
                'logger_cfgs': {
                    'save_model_freq': 25
                },
            }

            agent = omnisafe.Agent(alg, env_id)
            agent.learn()

            agent.plot(smooth=1)
            agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
            agent.evaluate(num_episodes=1)
