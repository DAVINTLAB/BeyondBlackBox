import sys
import os
sys.path.append(os.path.realpath('./libs/EVA02/'))

from templates.ObjectDetector import ObjectDetector
from EVA02 import load_model
from collections.abc import Mapping

import numpy as np
import cv2
import torch

import time

from matplotlib import pyplot as plt

class eva02(ObjectDetector):
    def __init__(self, args):
        super(eva02, self).__init__(args)
        self.model_args = args.__dict__
        self.threshold = args.threshold
        self.nms_threshold = .8

        self.original_image_size = None
        in_sz = 1024
        self.network_input_size = (in_sz, in_sz)
        self.total_prediction_time = 0.
    
    def __init_model__(self):
        class model_args:
            def __init__(self, **kwargs):
                for k,v in kwargs.items():
                    setattr(self, k, v)
        model_args = model_args(**self.model_args)
        self.model = load_model(model_args)

        
    def __pre_process__(self, images):
        ## Transform image to tensor and resize it to (1024,1024)
        data = []
        for image in images:
            if self.original_image_size is None:
                self.original_image_size = image.shape[:2]
            image = cv2.resize(image, self.network_input_size, interpolation=cv2.INTER_CUBIC)
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            data.append({"image": image})
        return data

    def __detect__(self, image):
        t0 = time.time()
        out = self.model(image)
        self.total_prediction_time+= time.time()-t0
        return out

    def __post_process__(self, net_output, causal_rules):
        remove_wrapper = lambda x : x.detach().cpu()
        clear_formatting = lambda x : [remove_wrapper(values).tolist() for values in x]
        fti = lambda x : [int(i) for i in x]
        out = []

        #attn_maps = self.model.backbone.net.blocks[-1].attn.attn_map

        for result in net_output:
            results = list(result.values())[0]
            tmp = []
    
            for i in range(len(results)):
                box = results.pred_boxes[i]
                score = remove_wrapper(results.scores[i]).item()
                label = remove_wrapper(results.pred_classes[i]).item()

                #attn_map = attn_maps[i]
                #attn_map-= attn_map.min()
                #attn_map/= attn_map.max()
                # attn_map = attn_map.reshape(64,64)
    
                box = clear_formatting(box)
                box = [
                    box[0][0] * self.original_image_size[1] / self.network_input_size[1],
                    box[0][1] * self.original_image_size[0] / self.network_input_size[0],
                    box[0][2] * self.original_image_size[1] / self.network_input_size[1],
                    box[0][3] * self.original_image_size[0] / self.network_input_size[0]
                ]
                tmp.append({
                    "bbox": fti(box), 
                    "score": score, 
                    "class": label, 
                    #"attn_map": attn_map.tolist()
                })
            out.append(tmp)
        return out

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
