## Wrapper class for object detector modules
import torch
import numpy as np
import random

# Prevent accidental re-seeding
def block_seed_setting(*args, **kwargs):
    print ("[WARNING] Seeding is disabled in this script.")

# Set seeds
import os
import time
ts = int(str(time.time()).split('.')[0][-7:])
torch.manual_seed(ts)
torch.cuda.manual_seed(ts)
torch.cuda.manual_seed_all(ts)
np.random.seed(ts)
random.seed(ts)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ["PYTHONHASHSEED"] = str(ts)
print ("[INFO]-Detector Seeding set to: ", ts)


# Monkey-patch seed-setting functions
torch.manual_seed = block_seed_setting
torch.cuda.manual_seed = block_seed_setting
torch.cuda.manual_seed_all = block_seed_setting
np.random.seed = block_seed_setting
random.seed = block_seed_setting
class ObjectDetector:
    def __init__(self, args):
        self.rules = None

    def assign_rules(self, rules):
        self.rules = rules

    def __pre_process__(self, image):
        raise NotImplementedError("Method not implemented")

    def __detect__(self, data):
        raise NotImplementedError("Method not implemented")

    def __post_process__(self, result, causal_rules):
        raise NotImplementedError("Method not implemented")

    def __apply_rules__(self, result, causal_rules):
        raise NotImplementedError("Method not implemented")

    def __output_assertions__(self, out):
        assert type(out) == list, "Output must be a list of lists of dictionaries"
        assert type(out[0]) == list, "Output must be a list of lists of dictionaries"
        out = out[0]
        
        if len(out) > 0:
            assert type(out[0]) == dict, "Output must be a list of lists of dictionaries"
            assert "bbox" in out[0], "Output dictionary must have 'bbox' key"
            assert "class" in out[0], "Output dictionary must have 'class' key"
            assert "score" in out[0], "Output dictionary must have 'score' key"
            assert "fid" in out[0], "Output dictionary must have 'fid' key"
            
            assert len(out[0]["bbox"]) == 4 and \
                    out[0]["bbox"][0] < out[0]["bbox"][2] and \
                    out[0]["bbox"][1] < out[0]["bbox"][3], \
                    "Bounding boxes must be in tlbr format"

            assert type(out[0]["class"]) == int, "Class must be an id"
            assert type(out[0]["score"]) == float and out[0]["score"] <= 1, "Score must be a float between 0 and 1"
            assert type(out[0]["bbox"][0]) == int, "Bounding box coordinates must be integers and correspond to position in original image size"


    def detect(self, image_data):
        if not len(image_data):
            return []

        if 'detections' in image_data[0]: ## if we already processed and are just doing ablation
            return self.__apply_rules__(image_data, self.rules)

        images = [obj['image'] for obj in image_data]
        ids = [obj['id'] for obj in image_data]

        data = self.__pre_process__(images)
        result = self.__detect__(data)

        out = self.__post_process__(result, self.rules)
        out = self.__apply_rules__(out, self.rules)

        for frame, fid in zip(out, ids):
            for obj in frame:
                obj["fid"] = fid

        self.__output_assertions__(out)

        for frame in out:
            for obj in frame:
                obj.pop("image", None) ## asserts that image is not carried forward here
                obj["h-results"] = {}  ## buffer for heuristic results throughout the pipeline

        return out
