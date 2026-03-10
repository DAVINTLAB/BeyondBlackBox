from templates.ObjectDetector import ObjectDetector

import numpy as np
import cv2
import torch

import os
import json
import time

from matplotlib import pyplot as plt

class coco(ObjectDetector):
    def __init__(self, args):
        super(coco, self).__init__(args)
        self.model_args = args.__dict__
        self.threshold = args.threshold
        self.nms_threshold = .8

        self.original_image_size = None
        in_sz = 1024
        self.network_input_size = (in_sz, in_sz)
        self.total_prediction_time = 0.
        self.fid = 0
    
    def __init_model__(self):
        file_name = os.path.basename(os.path.splitext(self.model_args['data_path'])[0])
        json_file = "{}.json".format(file_name)
        json_path = os.path.join(self.model_args['results_path'], json_file)

        if not os.path.exists(json_path):
            json_path = os.path.join(self.model_args['results_path'], f"stage#0_{json_file}")

        with open(json_path, 'r') as infile:
            tmp_results = json.load(infile)
        
        self.model_results = {}
        for entry in tmp_results:
            if entry['fid'] not in self.model_results:
                self.model_results[entry['fid']] = []
            self.model_results[entry['fid']].append(entry)
        
    def __pre_process__(self, images):
        return images

    def __detect__(self, image):
        results = []
        for img in image:
            tmp = []
            if self.fid in self.model_results:
                for entry in self.model_results[self.fid]:
                    tmp.append(entry)
            results.append(tmp)
            self.fid += 1
        return results


    def __post_process__(self, net_output, causal_rules):
        return net_output

    def __apply_rules__(self, net_output, causal_rules):
        ## convert data from dict of (x1,y1,x2,y2)to a list of (x,y,w,h) boxes and apply NMS
        results = []
        for data in net_output:
            boxes = [detection["bbox"][:] for detection in data]
            scores = [detection["score"] for detection in data]
    
            for box in boxes:
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
    
            indexes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.threshold, nms_threshold=self.nms_threshold)
            data = [data[i] for i in indexes]
            results.append(data)
        return results
