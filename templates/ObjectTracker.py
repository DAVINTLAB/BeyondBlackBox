from queue import Queue
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
print ("[INFO]-Tracker Seeding set to: ", ts)


# Monkey-patch seed-setting functions
torch.manual_seed = block_seed_setting
torch.cuda.manual_seed = block_seed_setting
torch.cuda.manual_seed_all = block_seed_setting
np.random.seed = block_seed_setting
random.seed = block_seed_setting

## Wrapper class for object tracker modules
class ObjectTracker:
    def __init__(self, args):
        self.buffer = Queue(maxsize=args.track_buffer)
        self.ignored_classes = args.ignored_classes
        self.ignored_buffer = Queue(maxsize=args.track_buffer)
        self.threshold = args.threshold
        self.prep_done = False
        self.done = False
        self.rules = None

    def assign_rules(self, rules):
        self.rules = rules

    def __pre_process__(self, image, detections):
        raise NotImplementedError("Method not implemented")

    def __track__(self, data, detections):
        raise NotImplementedError("Method not implemented")

    def __post_process__(self, result):
        raise NotImplementedError("Method not implemented")
    
    def __post_process_item__(self, result):
        raise NotImplementedError("Method not implemented")

    def __apply_rules__(self, data, ignored_data, causal_rules):
        raise NotImplementedError("Method not implemented")

    def __output_assertions__(self, tracked):
        assert type(tracked) == list, "Output must be a list of dictionaries"
        if len(tracked):
            assert type(tracked[0]) == dict, "Output must be a list of dictionaries"
            assert "bbox" in tracked[0], "Output dictionary must have 'bbox' key"
            assert "class" in tracked[0], "Output dictionary must have 'class' key"
            assert "score" in tracked[0], "Output dictionary must have 'score' key"
            assert "tid" in tracked[0], "Output dictionary must have 'tid' key"
            assert type(tracked[0]["tid"]) == int, "Track id must be an integer"
            assert type(tracked[0]["bbox"]) == list, "Bounding box must be a list"
            assert len(tracked[0]["bbox"]) == 4, "Bounding box must have 4 coordinates"
            assert type(tracked[0]["bbox"][0]) == int, "Bounding box coordinates must be integers"
            assert tracked[0]["bbox"][0] < tracked[0]["bbox"][2], "Bounding box coordinates must be in tlbr format"
            assert tracked[0]["bbox"][1] < tracked[0]["bbox"][3], "Bounding box coordinates must be in tlbr format"


    def track(self, image_data, detections):
        if image_data is None:
            self.done = True
            return

        image = image_data['image']
        frame_id = image_data['id']

        ignored, relevant = [], []
        det = {}
        for det in detections:
            if det['class'] in self.ignored_classes:
                ignored.append(det)
            else:
                relevant.append(det)
        already_processed = "tid" in det

        if already_processed:
            self.prep_done = True
            self.buffer.put(relevant)
            self.ignored_buffer.put(ignored)
            return

        data, dets = self.__pre_process__(image, relevant)
        track_data = self.__track__(data, dets, frame_id)
        tracked = self.__post_process__(track_data)

        self.buffer.put(tracked)
        self.ignored_buffer.put(ignored)
        if self.buffer.full():
            self.prep_done = True
        return

    def retrieve_track(self):
        if not self.prep_done:
            return False ## still not ready
        if self.prep_done and self.buffer.empty():
            return True ## done

        ## wait for next frame to be ready
        if not self.buffer.full() and not self.done:
            return False

        self.__apply_rules__(self.buffer.queue, self.ignored_buffer.queue, self.rules)

        out = self.buffer.get()
        ignored_dets = self.ignored_buffer.get()
        
        out = self.__post_process_item__(out)
        out = self.__join_output__(out, ignored_dets)

        self.__output_assertions__(out)
        return out

    def __join_output__(self, tracked, ignored):
        if not len(tracked):
            return tracked
        joined = tracked[:]
        for track in ignored:
            track['fid'] = tracked[0]['fid']
            track['tid'] = 0
            joined.append(track)
        return joined
