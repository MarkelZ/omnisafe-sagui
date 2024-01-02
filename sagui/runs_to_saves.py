import os
import json
import shutil


runs_path = './runs/'
saves_path = './saves/'

if not os.path.exists(saves_path):
    os.mkdir(saves_path)

with os.scandir(runs_path) as entries:
    # Open each directory in ./runs/
    # Directory name should have the form `Algo-{Env-v0}`
    for dir in entries:
        if not dir.is_dir():
            continue

        # Open each subdirectory
        # Should have the form `seed-num-date`
        for subdir in os.scandir(dir):
            if not subdir.is_dir():
                continue

            # Open the config file
            config_path = os.path.join(subdir.path, 'config.json')

            if not os.path.exists(config_path):
                print(f'What is this? {subdir.path}')
                continue

            # Get the exploration noise used
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
                noise = config_data['algo_cfgs']['exploration_noise']
                noise = str(noise).replace('.', ',')

            # Create the save directory for this experiment
            env_num = dir.name.split('-')[1][-1]
            exp_path = f'save_actnoise_{noise}_guide{env_num}'
            exp_path = os.path.join(saves_path, exp_path)

            if not os.path.exists(exp_path):
                os.mkdir(exp_path)

            # Copy config.json to the save directory
            shutil.copyfile(config_path, os.path.join(exp_path, 'config.json'))

            # Make torch save directory
            exp_torch_path = os.path.join(exp_path, 'torch_save')

            if not os.path.exists(exp_torch_path):
                os.mkdir(exp_torch_path)

            # Copy saved model to the new directory
            model = 'epoch-500.pt'
            model_path = os.path.join(subdir.path, f'torch_save/{model}')
            shutil.copyfile(model_path, os.path.join(exp_torch_path, model))
