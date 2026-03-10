import sys
import os
sys.path.append(os.path.realpath('./libs/'))


import json
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from templates.ObjectTracker import ObjectTracker
from tracker.mc_SMILEtrack import SMILEtrack


class smiletrack(ObjectTracker):
    def __init__(self, args):
        super(smiletrack, self).__init__(args)
        self.model_args = args.__dict__
        self.seen_tracks = {}
        cudnn.benchmark = True
        # self.min_appearance = 30
    
    def __init_model__(self):
        class model_args:
            def __init__(self, **kwargs):
                for k,v in kwargs.items():
                    setattr(self, k, v)

        self.model_args = model_args(**self.model_args)
        self.tracker = SMILEtrack(self.model_args, frame_rate=self.model_args.video_framerate)
        self.min_appearance = self.model_args.min_appearance

    def __pre_process__(self, image, detections):
        formated_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            s = detection['score']
            c = detection['class']
            
            if s > self.threshold:
                formated_detections.append([x1, y1, x2, y2, s, detection])

        return image, np.asarray(formated_detections)

    def __track__(self, frame, detections, frame_id):
        """
        frame should be a BGR numpy array WHC
        detections should be [[x,y,x,y,s,c],...] numpy array 
        """
        if not len(detections):
            return {}
        online_targets = self.tracker.update(detections, frame)
        detections_data = {}

        for idx, track in enumerate(online_targets):
            if track.tlwh[2] * track.tlwh[3] > self.tracker.args.min_box_area:
                bbx = []
                for i,coord in enumerate(track.tlbr):
                    bbx.append(max(int(coord), 0))

                detections_data[track.track_id] = track.cls
                detections_data[track.track_id]["tid"] = track.track_id
                detections_data[track.track_id]["bbox"] = bbx
                detections_data[track.track_id]["fid"] = frame_id
                detections_data[track.track_id]["score"] = float(track.score)

        return detections_data 

    def __post_process__(self, data):
        return data
    
    def __post_process_item__(self, data):
        ## move to list of dictionaries
        return list(data.values())

    def __apply_rules__(self, data, ignored_data, causal_rules):
        # return data
        cur_fid = float('inf')
        track_data = {}
        for frame_data in data:
            for track in frame_data.values():
                if track['tid'] not in track_data:
                    track_data[track['tid']] = []
                track_data[track['tid']].append(track)
                cur_fid = min(track['fid'], cur_fid)

        for tracks in track_data.values():
            track_corr = {}
            cls_hist = {}
            oldest = None
            valid_tracks = []

            for track in tracks:
                bb = track['bbox']
                conf = track['score']
                lbl = track['class']

                if track['fid'] == cur_fid:
                    oldest = track

                if lbl not in track_corr:
                    track_corr[lbl] = 0
                    cls_hist[lbl] = []

                if not track['valid'] and conf < 0.9:
                    conf = 0 ## Maybe lower it instead? Maybe cut by half?
                else:
                    valid_tracks.append(track)

                track_corr[lbl] += conf
                cls_hist[lbl].append(conf)

            if oldest is None:
                continue
            
            cls = max(track_corr, key=track_corr.get)
            threat_confirmed = (oldest['class'] == 0) and (oldest['tid'] in self.seen_tracks)
            threat_confirmed = False
            if oldest['tid'] not in self.seen_tracks:
                self.seen_tracks[oldest['tid']] = -1
            self.seen_tracks[oldest['tid']] += 1

            scores = sum(cls_hist.values(), [])
            avg_score = sum(scores) / len(scores)
            avg_score = np.median(scores)
            #print ("Average score for track #{0}: {1} (max {2})".format(oldest['tid'], avg_score, max(scores)))

            for track in tracks:
                track['h-results']['short track'] = {
                    "satisfied": int(len(tracks) + self.seen_tracks[oldest['tid']] >= self.min_appearance),
                    "details": {
                        "track length": len(tracks)+self.seen_tracks[oldest['tid']],
                        "min required": self.min_appearance,
                    }
                }

                track['h-results']['reliable track'] = {
                    "satisfied": int(avg_score > .5),
                    "details": {
                        "average score": avg_score,
                        "min score": min(scores),
                        "max score": max(scores),
                        "min average required": 0.5,
                    }
                }

                track['h-results']['consistent pose'] = {
                    "satisfied": int(len(valid_tracks) >= self.min_appearance/3 or max(scores) >= 0.9),
                    "details": {
                        "valid poses": len(valid_tracks)/len(scores), # len(valid_tracks),
                        "min required": self.min_appearance/3,
                        "max score": max(scores),
                    }
                }


            if (len(tracks) + self.seen_tracks[oldest['tid']] < self.min_appearance or avg_score < 0.5) and \
                    (max(scores) < 0.9 or len(valid_tracks) < self.min_appearance/3):
                for frame_data in data:
                    to_remove = []
                    for k, track in frame_data.items():
                        if track['tid'] == oldest['tid']:
                            to_remove.append(k)
                    #for k in to_remove:
                    #    del frame_data[k]

            else:
                for idx in range(2):
                    if idx < len(tracks):
                        if tracks[idx]['score'] < 0.9:
                            tracks[idx]['class'] = 0 if threat_confirmed else cls
        
        cur_frame_main_data = list(data[0].values())
        cur_frame_ign_data  = ignored_data[0]

        #print (cur_frame_main_data)
        proc_data = self.rules.apply_rules_tracker(cur_frame_main_data, cur_frame_ign_data)
        proc_data = {track['tid']: track for track in proc_data}
        #print (proc_data)
        data[0] = proc_data

        return data
