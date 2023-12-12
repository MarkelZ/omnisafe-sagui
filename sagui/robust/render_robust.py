import numpy as np
from evaluator_robust import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs

# LOG_DIR should contain two things:
# 1. config.json
# 2. torch_save/{MODEL_FNAME}
LOG_DIR = './save/'
MODEL_FNAME = 'epoch-500.pt'

# Coeffs
coef_dict = {'body_mass': 1, 'dof_damping': 0.6}

register_sagui_envs()
evaluator = EvaluatorRobust(render_mode='human')
evaluator.load_saved(save_dir=LOG_DIR, model_name=MODEL_FNAME)
evaluator.render(coef_dict=coef_dict, num_episodes=1)
