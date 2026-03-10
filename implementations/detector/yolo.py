import os
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
from ultralytics import YOLO
from templates.ObjectDetector import ObjectDetector

class yolo(ObjectDetector):
    def __init__(self, args):
        super(yolo, self).__init__(args)
        self.args = args
        self.total_prediction_time = 0.

    def __init_model__(self):
        self.model = YOLO(self.args.detector_model)
        
    def __pre_process__(self, image):
        return image

    def __detect__(self, image):
        tmp = self.model(image, imgsz=(1024,1024), iou=0.8, conf=0.1)
        return tmp

    def __post_process__(self, result, causal_rules):
        remove_wrapper = lambda x : x.detach().cpu()
        out = []
        for objs in result:
            tmp = []
            for obj in objs.boxes:
                obj_data = {}
                obj_data["bbox"] = remove_wrapper(obj.xyxy).tolist()[0]
                obj_data["bbox"] = list(map(int, obj_data["bbox"]))
                obj_data["class"] = int(remove_wrapper(obj.cls).item())
                obj_data["score"] = remove_wrapper(obj.conf).item()
                tmp.append(obj_data)
            out.append(tmp)
        return out

    def __apply_rules__(self, data, causal_rules):
        return data
