import sys
import os
sys.path.append(os.path.realpath('./libs/estimator/'))

from core.inference import get_final_preds
from templates.PoseEstimator import PoseEstimator

from torchvision.transforms import v2

import numpy as np 
import torch
import cv2


class Config:
    class TEST:
        BLUR_KERNEL = None

class transpose(PoseEstimator):
    def __init__(self, args):
        super(transpose, self).__init__(args)
        input_sz = args.estimator_model.split('_')[-1].split('x')
        input_sz = list(map(int, input_sz))
        self.input_size = input_sz ## (256,192)
        self.device = args.device
        self.model_name = args.estimator_model
        self.ksize = args.estimator_kernel_size
    
    def __init_model__(self):
        self.cfg = Config()
        self.cfg.TEST.BLUR_KERNEL = self.ksize
        self.transforms = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(size=self.input_size),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToTensor()
        ])

        self.tpr = torch.hub.load('yangsenius/TransPose:main', self.model_name #'tpr_a4_256x192'
                            , pretrained=True).to(self.device)
        self.tpr.eval()
        self.expand_bbox = True
        self.expand_bbox_ratio = 0.25


    def __pre_process__(self, image, detections):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for obj in detections:
            x1,y1,x2,y2 = obj['bbox']

            if self.expand_bbox:
                expand_x = (x2 - x1) * self.expand_bbox_ratio
                expand_y = (y2 - y1) * self.expand_bbox_ratio
                x1 = max(0, int(x1 - expand_x))
                x2 = min(image.shape[1], int(x2 + expand_x))
                y1 = max(0, int(y1 - expand_y))
                y2 = min(image.shape[0], int(y2 + expand_y))

            obj['img'] = image[y1:y2,x1:x2]

        return detections

    def __estimate__(self, data):
        for obj in data:
            img = obj['img']
            img = self.transforms(img)

            query_locations = self.inference(img)
            obj['pose'] = query_locations
        
            #assert (query_locations[:,0] >= 0).all() 
            #assert (query_locations[:,0] <= self.input_size[1]).all()
            #assert (query_locations[:,1] >= 0).all()
            #assert (query_locations[:,1] <= self.input_size[0]).all()

        return data

    def __post_process__(self, result):
        for obj in result:
            locations = obj['pose']
            x1,y1,x2,y2 = obj['bbox']
            original_shape = obj['img'].shape

            if self.expand_bbox:
                expand_x = (x2 - x1) * self.expand_bbox_ratio
                expand_y = (y2 - y1) * self.expand_bbox_ratio
                x1 = max(0, int(x1 - expand_x))
                x2 = min(original_shape[1], int(x2 + expand_x))
                y1 = max(0, int(y1 - expand_y))
                y2 = min(original_shape[0], int(y2 + expand_y))
        
            locations[:,1] = locations[:,1] * (original_shape[0] / self.input_size[0])
            locations[:,0] = locations[:,0] * (original_shape[1] / self.input_size[1])

            locations[:,0] = locations[:,0] + x1
            locations[:,1] = locations[:,1] + y1
            #assert (locations[:,0] >= x1).all() 
            #assert (locations[:,0] <= x2).all()
            #assert (locations[:,1] >= y1).all()
            #assert (locations[:,1] <= y2).all()

            obj['pose'] = locations.tolist()
            del obj['img']

        return result

    def inference(self, img):
        with torch.no_grad():
            inputs = torch.cat([img.to(self.device)]).unsqueeze(0)
            outputs = self.tpr(inputs).detach().cpu()
        
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
    
        preds, maxvals = get_final_preds(
                self.cfg, output.numpy(), None, None, transform_back=False)
        return np.array([p*4+0.5 for p in preds[0]])
    

    def __apply_rules__(self, objs_with_skeleton, objs_without_skeleton, causal_rules):
        objs_with_skeleton = causal_rules.apply_rules_estimator(objs_with_skeleton, objs_without_skeleton)

        return objs_with_skeleton
