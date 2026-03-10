from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.utils.env import seed_all_rng

import random
import numpy as np
import torch

# Define a function to set the seed
def set_seed(seed):
    random.seed(seed)               # Python random seed
    np.random.seed(seed)            # NumPy random seed
    torch.manual_seed(seed)         # PyTorch CPU seed
    torch.cuda.manual_seed(seed)    # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed) # If using multi-GPU
    # Additional configuration to ensure determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_all_rng(seed)              # Detectron2 seed
    print ('=====================================')
    print (f'Seed set to {seed}')
    print ('=====================================')

# Set the seed
#seed = 42  # Replace with your desired seed
#set_seed(seed)
def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def load_model(args):
	cfg = LazyConfig.load(args.config_file)
	cfg = LazyConfig.apply_overrides(cfg, args.opts)
	default_setup(cfg, args)

	model = instantiate(cfg.model)
	model.to(cfg.train.device)

	DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
	model.eval()
	enable_dropout(model)
	return model.half()
