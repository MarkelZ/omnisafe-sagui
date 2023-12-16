import numpy as np
from evaluator_robust import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs

# LOG_DIR should contain two things:
# 1. config.json
# 2. torch_save/{MODEL_FNAME}
LOG_DIR = './save/'
MODEL_FNAME = 'epoch-500.pt'

# Create a list of coefficients
coef_list = [{'body_mass': mass_mult, 'dof_damping': damp_mult}
             for mass_mult in np.linspace(0.25, 4, 10)
             for damp_mult in np.linspace(0.5, 1.5, 10)]

register_sagui_envs()
evaluator = EvaluatorRobust(render_mode='human')
evaluator.load_saved(save_dir=LOG_DIR, model_name=MODEL_FNAME)
res = evaluator.evaluate(coef_list, num_episodes=100)
res_str = '[\n' + ',\n'.join([str(x) for x in res]) + '\n]'

with open(LOG_DIR + 'robust_results.txt', 'w') as f:
    f.write(res_str)
