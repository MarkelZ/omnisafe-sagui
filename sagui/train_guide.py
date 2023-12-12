import omnisafe
from omnisafe.envs.sagui_envs import register_sagui_envs


if __name__ == '__main__':
    register_sagui_envs()
    env_id = 'SafetyPointGuide0-v0'

    cfgs = {
        'train_cfgs': {
            'torch_threads': 8
        },
        'logger_cfgs': {
            'save_model_freq': 25
        },
    }

    agent = omnisafe.Agent('DDPGLagUnfold', env_id)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
