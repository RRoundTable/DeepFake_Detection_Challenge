
def get_config():
    return {
        'data': 'total',
        'resume': False,
        'num_workers':4,
        'weights': './weights/convLSTM_wd_0_drop_None_epoch_4_acc_0.6000_prec1_0.7500_recall1_0.3000.pth',
        'train_iter': 1000,
        'model': 'fc',
        'batch_size':16,
        'lr': 1e-4,
        'weight_decay': 0,
        'gamma': 0.1,
        'step_size':7,
        'momentum':0.9,
        'dropout': 0.5,
        'optimizer': 'radam',
        'seq_len': 10,
        'mask': 0.1,
        'input_size': (200, 200),
        'input_dim': 3,
        'hidden_dim':10, 
        'model': 'convLSTM',
        'kernel_size': (9, 9), # NOTE: odd number
        'num_layers': 2,
    }


def get_aug_config():
    aug_cfg = {
        "cutout": {"apply": True, "n_mask": 50, "size": 200, "length": 15, "p": 0.7},
        "Shear": {'X':(-15, 15), 'Y': (-15, 15), 'p': 0},
        "Affine": {"translate_percent": (0.0, 0.1), "rotate": (0.0, 0.0), 'p': 0.5},
        "GaussianBlur": {"sigma": (0, 0.05), "p": 0},
        "LinearContrast": {"alpha": (0.75, 1.5)},
        "AdditiveGaussianNoise": {
            "loc": 0,
            "scale": (0.0, 0.05 * 255),
            "per_channel": 0.5,
            'p': 0.5
        },
        "Multiply": {"mul": (0.8, 1.2), "per_channel": 0.2},
    }
    return aug_cfg