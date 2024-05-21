import os
import yaml


def save_config_file(full_config):
    del full_config['model_config']['datatype']
    full_config['model_config']['datatype'] = str(full_config['model_config']['datatype'])
    file = open(os.path.join(full_config['model_config'].save_checkpoint, 'config.yaml'), "w")
    yaml.dump(full_config, file)
    file.close()


def plot_loss():
    pass