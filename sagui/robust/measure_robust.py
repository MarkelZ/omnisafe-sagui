import numpy as np
from evaluator_robust import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs

# LOG_DIR should contain two things:
# 1. config.json
# 2. torch_save/{MODEL_FNAME}
LOG_DIR = './save/'
MODEL_FNAME = 'epoch-500.pt'

# Create a list of coefficients
coef_list = []
for mass in np.linspace(1e-6, 0.01, 16):
    for fric in np.linspace(0, 0.01, 8):
        coef_dic = {'body_mass': mass, 'dof_frictionloss': fric}
        coef_list.append(coef_dic)

register_sagui_envs()
evaluator = EvaluatorRobust(render_mode='human')
evaluator.load_saved(save_dir=LOG_DIR, model_name=MODEL_FNAME)
res = evaluator.evaluate(coef_list, num_episodes=100)
res_str = '[\n' + ',\n'.join([str(x) for x in res]) + '\n]'

with open(LOG_DIR + 'robust_results.txt', 'w') as f:
    f.write(res_str)
