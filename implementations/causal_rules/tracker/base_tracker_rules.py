import numpy as np
import math
from copy import deepcopy

MATCH_HISTORY = {}

def apply_rules(objs_with_skeleton, objs_without_skeleton):
    # Rule 1: Every object without a skeleton should be within half/twothirds -- half seems to work better
    #         the distance of elbow-wrist of at least one wrist (serving as a "grab range")
    #         If it is, then it is matched with the skeleton that has the forearm best aligned to it
    #         If it is not, then it is discarded -> unmatched objects may be missdetections!
    estimate_owner(objs_with_skeleton, objs_without_skeleton)
    look_for_missing_matches(objs_with_skeleton, objs_without_skeleton)

    return objs_with_skeleton

def look_for_missing_matches(objs_with_skeleton, objs_without_skeleton):
    #print ("[DEBUG] Looking for missing matches")
    for skel in objs_with_skeleton:
        if skel['class'] != 0:
            continue

        tid = skel['tid']

        #print ("[DEBUG] Checking skeleton (", tid, ") Asserting that has a match (", 'match' in skel, ") and in Match history (", tid in MATCH_HISTORY, ")")

        if 'match' not in skel and tid in MATCH_HISTORY:
            for which_hand, (dist, prev_wrist, prev_forearm_vector, prev_obj) in MATCH_HISTORY[tid].items():
                bbox = prev_obj['bbox']
                if prev_obj['score'] < .4:
                    continue

                curr_elbow, curr_wrist, curr_forearm_vector, distance = get_hand_information(skel, which_hand)
                if curr_elbow is None:
                    continue


                new_distance, estimated_bbox = estimate_object_position(prev_wrist, prev_forearm_vector, bbox, dist, curr_wrist, curr_forearm_vector)

                estimated_object = deepcopy(prev_obj)
                estimated_object['bbox'] = estimated_bbox
                estimated_object['score'] *= .9
                estimated_object['fid'] = skel['fid']
                estimated_object['h-results'] = {'inferred' : 1}

                skel['match'] = len(objs_without_skeleton)
                # print ("Inserting missing match for skeleton", tid, "with object", len(objs_without_skeleton))
                objs_without_skeleton.append(estimated_object)
                MATCH_HISTORY[tid][which_hand] = (new_distance, curr_wrist, curr_forearm_vector, estimated_object)
            
def estimate_object_position(wrist_prev, forearm_prev, bounding_box, distance, wrist_curr, forearm_curr):
    # Unpack the bounding box
    xc = (bounding_box[0] + bounding_box[2]) / 2
    yc = (bounding_box[1] + bounding_box[3]) / 2
    center = np.array([xc, yc])
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]

    # Calculate scale factor and angle difference
    scale_factor = np.linalg.norm(forearm_curr) / np.linalg.norm(forearm_prev)
    forearm_angle_diff = math.radians(calculate_angle(forearm_prev, forearm_curr))

    # Update bounding box size and distance
    new_width = width * scale_factor
    new_height = height * scale_factor
    new_distance = distance * scale_factor

    # Compute object vector in previous frame relative to wrist
    wrist_object_vector = (center - wrist_prev)
    wrist_object_vector = (wrist_object_vector / np.linalg.norm(wrist_object_vector)) * new_distance

    # Rotate the vector by the forearm angle difference
    rotation_matrix = np.array([
        [math.cos(forearm_angle_diff), -math.sin(forearm_angle_diff)],
        [math.sin(forearm_angle_diff), math.cos(forearm_angle_diff)]
    ])
    rotated_vector = rotation_matrix @ wrist_object_vector

    # Calculate the new object position
    object_center_curr = wrist_curr + rotated_vector
    object_curr = [
        object_center_curr[0] - new_width / 2,
        object_center_curr[1] - new_height / 2,
        object_center_curr[0] + new_width / 2,
        object_center_curr[1] + new_height / 2
    ]

    return new_distance, list(map(int, object_curr))

def estimate_owner(objs_with_skeleton, objs_without_skeleton, ANGLE_THRESHOLD=45):
    MATCH_HISTORY.clear()
    potential_matches = {}
    for oid, obj in enumerate(objs_without_skeleton):
        potential_matches[oid] = (float('inf'), float('inf'), None, None, None, None)

    for sid, skel in enumerate(objs_with_skeleton):
        #if skel['class'] != 0:
        #    continue

        e1, w1, wrist1_vector, dist_wrist1 = get_hand_information(skel, 'left')
        e2, w2, wrist2_vector, dist_wrist2 = get_hand_information(skel, 'right')

        #distances = [(w1, wrist1_vector, dist_wrist1/2), (w2, wrist2_vector, dist_wrist2/2)]
        distances = [(w1, wrist1_vector, dist_wrist1, 'left'), (w2, wrist2_vector, dist_wrist2, 'right')]
        #distances = [(w1, 2*dist_wrist1/3), (w2, 2*dist_wrist2/3)]

        for oid, obj in enumerate(objs_without_skeleton):
            x1,y1,x2,y2 = obj['bbox']
            center = np.asarray([(x1+x2)/2, (y1+y2)/2])
            #points = np.asarray([(x1,y1), (x2,y1), (x2,y2), (x1,y2), center])
            points = np.asarray([center])

            for wrist, w_e_vector, grab_range, which_hand in distances:
                if wrist is None:
                    continue
                # dist_to_hand = np.linalg.norm(center - wrist)
                dist_to_hand = np.min(np.linalg.norm(points - wrist, axis=1))

                if dist_to_hand <= grab_range:
                    w_o_vector = center - wrist
                    angle = calculate_angle(w_e_vector, w_o_vector)
                    if angle < ANGLE_THRESHOLD:
                        potential_matches[oid] = min(potential_matches[oid], (angle, dist_to_hand, sid, wrist, w_e_vector, which_hand))

    # removes multiple matches to the same wrist
    to_remove = set([])
    for oid1, (angle1, dist1, sid1, wrist1, forearm_vector, which_hand) in potential_matches.items():
        for oid2, (angle2, dist2, sid2, wrist2, forearm_vector, which_hand) in potential_matches.items():
            if sid1 is None or sid2 is None:
                continue
            if oid1 != oid2 and (wrist1 == wrist2).all():
                # conf1 = (1-objs_without_skeleton[oid1]['score'])*angle1
                # conf2 = (1-objs_without_skeleton[oid2]['score'])*angle2
                conf1 = objs_without_skeleton[oid1]['score']*(360-angle1)*dist1
                conf2 = objs_without_skeleton[oid2]['score']*(360-angle2)*dist2

                # if conf1 < conf2:
                if conf1 > conf2:
                    del_id = oid2
                else:
                    del_id = oid1
                to_remove.add(del_id)


    for oid, (angle, dist, sid, wrist, forearm_vector, which_hand) in list(reversed(potential_matches.items())):
        objs_without_skeleton[oid]['h-results']["matching skeleton"] = {
            "satisfied": int(sid is not None),
            "details": {
                "angle": angle,
                "distance": dist,
            }
        }
        objs_without_skeleton[oid]['h-results']["best match"] = {
            "satisfied": int(oid not in to_remove),
            "details": {
                "match score": objs_without_skeleton[oid]['score'] * (360-angle) * dist
            }
        }

        if sid is None or oid in to_remove:
            # if objs_without_skeleton[oid]['score'] < .9:
            #     del objs_without_skeleton[oid]
            del potential_matches[oid]
            #objs_without_skeleton[oid]['match'] = 0
        else:
            ## What if matching with the wrong skeleton?
            objs_with_skeleton[sid]['match'] = oid
            objs_without_skeleton[oid]['match'] = objs_with_skeleton[sid]['tid']
            
            tid = objs_with_skeleton[sid]['tid']
            if tid not in MATCH_HISTORY:
                MATCH_HISTORY[tid] = {}
            MATCH_HISTORY[tid][which_hand] = (dist, wrist, forearm_vector, objs_without_skeleton[oid])


def get_hand_information(skeleton, which_hand):    
    #Given an skeleton S
    # S[1] = center of the head
    # S[5] = Shoulder1
    # S[6] = Shoulder2
    # S[7] = Elbow1
    # S[8] = Elbow2
    # S[9] = Wrist1
    # S[10] = Wrist2

    LEFT_WRIST = 9
    LEFT_ELBOW = 7
    LEFT_SHOULDER = 5

    RIGHT_WRIST = 10
    RIGHT_ELBOW = 8
    RIGHT_SHOULDER = 6

    if which_hand == 'left':
        w = np.asarray(skeleton['pose'][LEFT_WRIST])
        e = np.asarray(skeleton['pose'][LEFT_ELBOW])
        s = np.asarray(skeleton['pose'][LEFT_SHOULDER])
        wc = np.asarray(skeleton['pose_score'][LEFT_WRIST])
        ec = np.asarray(skeleton['pose_score'][LEFT_ELBOW])
        sc = np.asarray(skeleton['pose_score'][LEFT_SHOULDER])
        wv = np.asarray(skeleton['pose_visibility'][LEFT_WRIST])
    else:
        w = np.asarray(skeleton['pose'][RIGHT_WRIST])
        e = np.asarray(skeleton['pose'][RIGHT_ELBOW])
        s = np.asarray(skeleton['pose'][RIGHT_SHOULDER])
        wc = np.asarray(skeleton['pose_score'][RIGHT_WRIST])
        ec = np.asarray(skeleton['pose_score'][RIGHT_ELBOW])
        sc = np.asarray(skeleton['pose_score'][RIGHT_SHOULDER])
        wv = np.asarray(skeleton['pose_visibility'][RIGHT_WRIST])
            
    e_w_vector = w - e
    e_s_vector = s - e

    # print (wv, wc)

    if wc < .6 or ec < .6 or sc < .6 or wv < .6:
        return None, None, None, None

    dist_wrist = np.linalg.norm(e_w_vector)
    dist_shoulder = np.linalg.norm(e_s_vector)

    if dist_wrist < dist_shoulder:
        dist_wrist = dist_shoulder

    ## If the distance between the elbow and the wrist is too small, then the wrist is not considered
    ## as a valid point, as it may be cut off by the bounding box detected
    ## In this case, use the distance between the elbow and the shoulder to estimate where the wrist is
    if dist_wrist < dist_shoulder * .2 and False:
        orig_wrist = w
        norm = e_w_vector / dist_wrist
        w = e + norm * dist_shoulder
        e_w_vector = e_s_vector
        dist_wrist = dist_shoulder
        print ("Wrist too close to elbow, using shoulder-elbow distance to estimate wrist location")
        print ("Skeleton Wrist:", orig_wrist, "Estimated Wrist:", w)

    return e, w, e_w_vector, dist_wrist


def calculate_angle(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for numerical stability
    angle_deg = np.degrees(angle_rad)
    return angle_deg
