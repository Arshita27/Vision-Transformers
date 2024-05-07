import ml_collections
import torch


def get_params_and_config():
    _DATA_CONFIG = {
    'training_data': 'imagenet',
    'training_data_dir':None,
    'patch_size': 32,
    'patch_stride': 32,
    }

    # Model backbone config.
    _MODEL_CONFIG = {
        'use_backbone': True,
        'hidden_size': 384,
        'representation_size': None,
        'resnet_emb': {
            'num_layers': 5
        },
        'transformer': {
            'attention_dropout_rate': 0,
            'dropout_rate': 0,
            'mlp_dim': 1152,
            'num_heads': 6,
            'num_layers': 14,
            'use_sinusoid_pos_emb': False,
            'drop_path': True
        },
        'weight_standardization': True,
        'dropout_path':True,
        'dropout_path_rate':'constant', #  or 'linear'
        'train_batchsize': 64,
        'test_batchsize': 64,
        'n_classes': 1000,
        'data_type': torch.float,
        'gpu_id': "0",
        'n_epochs': 50,
        'learning_rate': 0.0001,
        'optimizer': 'adamw', # or adam
        'weight_decay':0.00001,
        'momentum': 0.9,
        'T_max': 10000,
        'eta_min': 0.0,
        'save_freq': 1,
        'val_freq':5,
        'save_checkpoint':'ckpts/',
        'load_pretrained_checkpoints': None,
    }
    return ml_collections.ConfigDict(_DATA_CONFIG), ml_collections.ConfigDict(_MODEL_CONFIG)