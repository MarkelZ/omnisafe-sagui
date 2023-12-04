
# Hyperparameters from sagui paper Appendix G, Static task.
cfgs = {
    'train_cfgs': {
        'total_steps': 1500000,
    },
    'algo_cfgs': {
        'alpha': 1,
        'cost_normalize': False,
        'gamma': 0.99,
        'steps_per_epoch': 30000,
        'update_iters': 100,
        'batch_size': 32,
        'size': 1000000,  # Replay buffer
        'start_learning_steps': 500,
        'warmup_epochs': 1,
    },
    'model_cfgs': {
        'actor': {
            'hidden_sizes': [32, 32],
            'lr': 1e-3,
        },
        'critic': {
            'hidden_sizes': [32, 32],
            'lr': 1e-3,
        }
    },
    'lagrange_cfgs': {
        'cost_limit': 5.0,
        'lagrangian_multiplier_init': 0.000,
        'lambda_lr': 50 * 1e-3,
        'lambda_optimizer': 'Adam',
    },
    'logger_cfgs': {
        'save_model_freq': 25,
    }
}
