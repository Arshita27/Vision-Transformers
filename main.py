import ml_collections

from default_config import get_params_and_config
from models import model as m


data_config, model_config = get_params_and_config()
cfg = ml_collections.ConfigDict({'data_config': data_config, 'model_config':model_config})
print(cfg)

model = m.Model(cfg)
print(model)