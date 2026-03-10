import cv2
import numpy as np

# class to iterate over video frames
class video:
    def __init__(self, args):
        self.cap = cv2.VideoCapture(args.data_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.capture_rate = 1 # args.capture_rate
        self.std_size = (768,480)
        self.fid = 0

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        for _ in range(self.capture_rate-1):
            self.cap.grab()

        if not ret:
            raise StopIteration

        frame = cv2.resize(frame, self.std_size, interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret = {"image": frame, "id": self.fid}
        self.fid += self.capture_rate

        return ret

    def __del__(self):
        self.cap.release()
