from my_evaluator import MyEvaluator
from omnisafe.envs.sagui_envs import register_sagui_envs

# LOG_DIR should contain two things:
# 1. config.json
# 2. torch_save/{MODEL_FNAME}
LOG_DIR = './save_oldsagui/'
MODEL_FNAME = 'epoch-500.pt'

register_sagui_envs()
evaluator = MyEvaluator()
evaluator.load_saved(save_dir=LOG_DIR, model_name=MODEL_FNAME, render_mode='rgb_array',
                     camera_name='track', width=256, height=256)
evaluator.render(num_episodes=1, deterministic=False)
