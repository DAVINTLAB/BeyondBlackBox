import numpy as np
import math
from copy import deepcopy

MATCH_HISTORY = {}

def apply_rules(objs_with_skeleton, objs_without_skeleton):
    # Rule 1: A relevant pose is defined as having a head and at 
    #         least one arm, where any of these regions should 
    #         have a keypoint with a minimal score
    relevant_pose(objs_with_skeleton)

    return objs_with_skeleton

def relevant_pose(objs_with_skeleton):
    for skel in objs_with_skeleton:
        head = skel['pose_score'][:5]
        left_arm = skel['pose_score'][5:8]
        right_arm = skel['pose_score'][8:11]

        valid_head = any([score > 0.5 for score in head]) 
        valid_left_arm = any([score > 0.5 for score in left_arm]) 
        valid_right_arm = any([score > 0.5 for score in right_arm])

        valid_pose = valid_head and (valid_left_arm or valid_right_arm) \
                        or np.max(left_arm) + np.max(right_arm) > 1.4

        if not valid_pose and skel['score'] < 0.9:
            skel['valid'] = False
        else:
            skel['valid'] = True

        skel['h-results']["valid pose"] = {
            "satisfied": int(valid_pose), 
            "details": {
                "head_score": np.max(head),
                "left_arm_score": np.max(left_arm),
                "right_arm_score": np.max(right_arm)
            }
        }
