## Wrapper class for pose estimator modules
import torch
import numpy as np
import random

# Prevent accidental re-seeding
def block_seed_setting(*args, **kwargs):
    print ("[WARNING] Seeding is disabled in this script.")

# Set seeds
import time
ts = int(str(time.time()).split('.')[0][-7:])
torch.manual_seed(ts)
torch.cuda.manual_seed(ts)
torch.cuda.manual_seed_all(ts)
np.random.seed(ts)
random.seed(ts)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
print ("[INFO]-Estimator Seeding set to: ", ts)

# Monkey-patch seed-setting functions
torch.manual_seed = block_seed_setting
torch.cuda.manual_seed = block_seed_setting
torch.cuda.manual_seed_all = block_seed_setting
np.random.seed = block_seed_setting
random.seed = block_seed_setting

class PoseEstimator:
    def __init__(self, args):
        self.ignored_classes = args.ignored_classes
        self.rules = None
    
    def assign_rules(self, rules):
        self.rules = rules

    def __pre_process__(self, image, detections):
        raise NotImplementedError("Method not implemented")

    def __estimate__(self, data):
        raise NotImplementedError("Method not implemented")

    def __post_process__(self, result):
        raise NotImplementedError("Method not implemented")

    def __apply_rules__(self, result, causal_rules):
        raise NotImplementedError("Method not implemented")

    def __output_assertions__(self, result):
        assert type(result) == list, "Output must be a list of dictionaries"
        if len(result) > 0:
            assert type(result[0]) == dict, "Output must be a list of dictionaries"
            assert "pose" in result[0], "Output dictionary must have 'pose' key"
            assert type(result[0]["pose"]) == list, "Pose must be a list"
            #assert len(result[0]["pose"]) > 0, "Pose must have at least one keypoint"
            if len(result[0]["pose"]) > 0:
                assert type(result[0]["pose"][0]) == list, "Keypoint must be a list"
                assert len(result[0]["pose"][0]) == 2, "Keypoint must have 2 coordinates"
                assert type(result[0]["pose"][0][0]) == float, "Keypoint coordinates must be floats"
                assert type(result[0]["pose"][0][1]) == float, "Keypoint coordinates must be floats"

    def estimate(self, image_data, detections):
        image = image_data['image']
        out = []

        ignored, relevant = [], []
        det = {}
        for det in detections:
            if det['class'] in self.ignored_classes:
                ignored.append(det)
            else:
                relevant.append(det)
        already_processed = "pose" in det

        if already_processed:
            out = relevant
        elif len(relevant):
            data = self.__pre_process__(image, relevant)
            pose = self.__estimate__(data)
            out = self.__post_process__(pose)

        out = self.__apply_rules__(out, ignored, self.rules)
        out = self.__join_ignored__(out, ignored)

        self.__output_assertions__(out)
        return out

    def __join_ignored__(self, out, ignored):
        joined = []
        if len(out):
            joined = out[:]

        for obj in ignored:
            obj['pose'] = []
            joined.append(obj)
        return joined
