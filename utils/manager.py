## uses threads to run modules independently
from threading import Thread
# from queue import Queue
import time
import traceback

import implementations
from copy import deepcopy

## Alternative to deal with incompatible library versions
from multiprocessing import Process, Queue, set_start_method, Barrier
import sys
import os


## DEBUGGING
from datetime import datetime

def check_seed(whom):
    import torch
    import numpy as np
    import random
    print (f"[DEBUG]-{whom}", torch.initial_seed(), torch.randint(0, 1000, (1,)))
    print (f"[DEBUG]-{whom}", torch.cuda.initial_seed(), torch.randint(0, 1000, (1,), device="cuda"))
    print(f"[DEBUG]-{whom}", np.random.get_state()[1][0], np.random.randint(0, 1000, (1,)))
    print(f"[DEBUG]-{whom}", random.getstate()[1][0], random.randint(0, 1000))

#####

def change_env(name):
    env = os.path.realpath(f'./envs/{name}')
    pyver = os.listdir(os.path.join(env, 'lib'))[0]
    for i in range(len(sys.path), 0, -1):
        if 'packages' in sys.path[i-1]:
            del sys.path[i-1]
    sys.path.insert(0, os.path.join(env, 'lib', pyver, 'site-packages'))
    sys.path.insert(0, os.path.join(env, 'bin'))
    print (sys.path)

## Class that manages and runs all modules
class Manager:
    def __init__(self, args, queue_size=2):
        ## To initialize processes with CUDA
        set_start_method('spawn')
        ###
        self.return_intermediate = args.return_intermediate
        self.detector_batch_size = queue_size
        self.__finished__ = Queue()

        self.detector_inputs = Queue(queue_size)
        self.estimator_inputs = Queue()
        self.tracker_inputs = Queue()
        self.outputs = Queue()

        self.sync = Barrier(3)
        self.err_queue = Queue()

        detector_thread = Process(target=self.__setup_detector__, args=(args,))
        tracker_thread = Process(target=self.__setup_tracker__, args=(args,))
        estimator_thread = Process(target=self.__setup_estimator__, args=(args,))

        detector_thread.start()
        tracker_thread.start()
        estimator_thread.start()

        self.detector_thread = detector_thread
        self.tracker_thread = tracker_thread
        self.estimator_thread = estimator_thread

        Thread(target=self.__check_errors__).start()

    def __setup_detector__(self, args):
        import sys
        import io
        import torch

        print ("Detector running")
        self.detector = implementations.load('detector', args.detector, args)
        causal_rules = None ##TODO:: Implement causal rules

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.detector.__init_model__()
        finally:
            sys.stdout = original_stdout

        try:
            with torch.no_grad():
                self.__run_detector__()
        except Exception as e:
            traceback.print_exc()
            self.err_queue.put(e)
        self.sync.wait()
        print ("Detector Passed!")
        check_seed("Detector")

    def __run_detector__(self):
        time_spent = 0.
        max_batch_size = 0
        finished = False
        started = False
        while True:
            inputs = []
            previous_stages = []
            for i in range(self.detector_batch_size):
                try:
                    # print ("Detector data remaining:", self.detector_inputs.qsize())
                    frame_data, prev_results = self.detector_inputs.get(block=True, timeout=.025)
                    if not started:
                        started = True
                        print ("Detector started, ts: {0}".format(datetime.now()))

                    if frame_data is None:
                        finished = True
                        break
                    inputs.append(frame_data)
                    previous_stages.append(prev_results)

                except Exception as e:
                    traceback.print_exc()
                    print (self.detector_inputs.qsize())
                    break

            if not len(inputs):
                if finished:
                    break
                continue

            max_batch_size = max(max_batch_size, len(inputs))

            t0 = time.time()
            SKIP = False
            NSKIP = 2
            if SKIP:
                half_inputs = [inputs[i] for i in range(len(inputs)) if i % NSKIP == 0]
                half_boxes = self.detector.detect(half_inputs)
                boxes = []
                for box in half_boxes:
                    boxes.append(box)
                    boxes.append(box)
                if len(inputs) % NSKIP:
                    boxes.pop()
            else:
                boxes = self.detector.detect(inputs)
            time_spent += time.time() - t0
            #print ("Detector sending data, remaining {0}".format(self.detector_inputs.qsize()))

            for frame_data, bxs, prev in zip(inputs, boxes, previous_stages):
                self.estimator_inputs.put((frame_data, prev+[bxs]))

            if finished:
                break
        print (">>>>>>>>>>>>Detector is Done<<<<<<<<<<<<<<<<<")
        print ("Detector conclusion ts: {0}".format(datetime.now()))
        print ("Detector time spent: {0}".format(time_spent))
        print ("Max batch size: {0}".format(max_batch_size))
        print ("Network time spent: {0}".format(self.detector.total_prediction_time))
        self.estimator_inputs.put((None, [None]))
    
    def __setup_tracker__(self, args):
        print("Tracker running")
        self.tracker = implementations.load('tracker', args.tracker, args)
        causal_rules = implementations.load('causal_rules', args.causal_rules, args)
        self.tracker.__init_model__()
        self.tracker.assign_rules(causal_rules)

        self.first_send = None
        self.first_recv = None

        frame_datas = []
        producer_thread = Thread(target=self.__run_tracker_producer__, args=(frame_datas,))
        consumer_thread = Thread(target=self.__run_tracker_consumer__, args=(frame_datas,))

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        self.sync.wait()
        print ("Tracker Passed!")
        print ("Tracker overheard: {0}".format(self.first_recv - self.first_send))
        check_seed("Tracker")

    def __run_tracker_producer__(self, frame_datas):
        started = False
        try:
            time_spent = 0.
            while True:
                frame_data, prev_results = self.tracker_inputs.get()
                if not started:
                    started = True
                    print ("Tracker started, ts: {0}".format(datetime.now()))
                boxes = prev_results[-1]
                frame_datas.append((frame_data, prev_results))

                if self.return_intermediate:
                    boxes = deepcopy(boxes)

                t0 = time.time()
                tracks = self.tracker.track(frame_data, boxes)
                time_spent += time.time() - t0

                if self.first_send is None:
                    self.first_send = time.time()

                if frame_data is None:
                    print (">>>>>>>>>>>>Tracker producer is Done<<<<<<<<<<<<<<<<<")
                    print ("Producer conclusion ts: {0}".format(datetime.now()))
                    print ("Tracker producer time spent: {0}".format(time_spent))
                    break

        except Exception as e:
            traceback.print_exc()
            self.err_queue.put(e)
        
    def __run_tracker_consumer__(self, frame_datas):
        try:
            while True:
                tracks = self.tracker.retrieve_track()
                if len(frame_datas):
                    if frame_datas[0][0] is None:
                        print (">>>>>>>>>>>>Tracker consumer is Done<<<<<<<<<<<<<<<<<")
                        print ("Consumer conclusion ts: {0}".format(datetime.now()))
                        break

                if tracks is False:
                    time.sleep(.001)
                    continue 

                if self.first_recv is None:
                    print ("Tracker consumer started, ts: {0}".format(datetime.now()))
                    self.first_recv = time.time()

                frame_data, prev_results = frame_datas.pop(0)
                # print ("Tracker sending data, remaining {0}".format(len(frame_datas)))
                self.outputs.put((frame_data, prev_results+[tracks]))

        except Exception as e:
            traceback.print_exc()
            self.err_queue.put(e)
        self.outputs.put((None, [None]))
        

    def __setup_estimator__(self, args):
        print ("Changing Estimator's Environment")
        change_env(args.estimator)
        self.estimator = implementations.load('estimator', args.estimator, args)
        causal_rules = implementations.load('causal_rules', args.causal_rules, args)
        print ("Estimator running")
        self.estimator.__init_model__()
        self.estimator.assign_rules(causal_rules)

        import torch
        from torch.amp import autocast

        try:
            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    self.__run_estimator__()
        except Exception as e:
            traceback.print_exc()
            self.err_queue.put(e)
        self.sync.wait()
        self.__finished__.put(True)
        print ("Estimator Passed!")
        check_seed("Estimator")

    def __run_estimator__(self):
        time_spent = 0.
        started = False
        while True:
            frame_data, prev_results = self.estimator_inputs.get()

            if not started:
                started = True
                print ("Estimator started, ts: {0}".format(datetime.now()))

            if frame_data is None:
                break

            tracks = prev_results[-1]
            if self.return_intermediate:
                tracks = deepcopy(tracks)

            t0 = time.time()
            estimations = self.estimator.estimate(frame_data, tracks)
            time_spent += time.time() - t0

            self.tracker_inputs.put((frame_data, prev_results+[estimations]))

        print (">>>>>>>>>>>>Estimator is Done<<<<<<<<<<<<<<<<<")
        print ("Estimator conclusion ts: {0}".format(datetime.now()))
        print ("Estimator time spent: {0}".format(time_spent))
        self.tracker_inputs.put((None, [None]))

    def run(self, frame_data):
        self.detector_inputs.put((frame_data, []))
    
    def __check_errors__(self):
        if self.err_queue.qsize():
            self.detector_thread.terminate()
            self.tracker_thread.terminate()
            self.estimator_thread.terminate()
            os._exit(1)
        time.sleep(1)
        if not self.__finished__.qsize():
            Thread(target=self.__check_errors__).start()
        else:
            print ("Checker finished")


    def get(self):
        out = self.outputs.get()[1]
        if self.return_intermediate:
            return out
        return [out[-1]]
