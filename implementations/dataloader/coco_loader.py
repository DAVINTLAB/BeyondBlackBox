import os
import cv2
import json

class coco_loader:
    def __init__(self, args):
        with open(args.data_path, 'r') as infile:
            self.data = json.load(infile)

        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']

        self.__return_annots__ = False
        self.prefix = args.data_prefix
        self.data_shape = args.data_shape
    
    def __iter__(self):
        self.__index__ = 0
        return self
    
    def __next__(self):
        ## check if we have reached the end of the dataset
        if self.__index__ >= len(self.images):
            raise StopIteration

        ## load the objects for the image and its annotations
        image = self.images[self.__index__]
        
        ## load the image
        img = cv2.imread(os.path.join(self.prefix, image['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## IF not in 1280x720, resize (REMOVE IF NOT NEEDED)
        if self.data_shape is not None:
            if img.shape[1] != self.data_shape[0] or img.shape[0] != self.data_shape[1]:
                img = cv2.resize(img, self.data_shape)
        
        ## prepare the image object
        obj = {"image": img, "id": image['id']}
        if self.__return_annots__:
            annotations = [ann for ann in self.annotations if ann['image_id'] == image['id']]
            obj['annotations'] = annotations

        ## increment the index and return the image object
        self.__index__ += 1
        return obj
