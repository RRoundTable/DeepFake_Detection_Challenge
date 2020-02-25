

def get_config():
    return {
        'model': 'max-pooing',
        'batch_size':2,
        'lr': 1e-4,
        'weight_decay': 0,
        'gamma': 0.1,
        'step_size':7,
        'momentum':0.9,
        'dropout': None,
        'optimizer': 'sgd',
        'seq_len': 25,
        'mask': 0.1,
        'input_size': (200, 200),
        'input_dim': 3,
        'hidden_dim':20, 
        'model': 'convLSTM',
        'kernel_size': (11, 11), # NOTE: odd number
        'num_layers': 2,
    }