import importlib

def load(module_type, module_name, args):
    config = importlib.import_module(f'configs.{module_type}.{module_name}', __name__)
    ## merge argparser object with config file
    for k,v in args.__dict__.items():
        config.__setattr__(k, v)
    return importlib.import_module(f'.{module_type}.{module_name}', __name__).__getattribute__(module_name)(config)
