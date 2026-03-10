import implementations
from . import manager

from threading import Thread

import time ##TODO:: Remove this, only for benchmarking

## To properly import githubs used in the code in case they weren't prepared to be imported
#import sys
#import os
#sys.path.append(os.path.realpath('./libs/estimator/'))
#sys.path.append(os.path.realpath('./libs/EVA02/'))
#sys.path.append(os.path.realpath('./libs/'))
#os.environ['YOLO_VERBOSE'] = 'False'
##

def run(args):
    ## Setup dataloader
    dataloader = implementations.load('dataloader', args.dataloader, args)

    ## Setup manager to load modules
    execution_manager = manager.Manager(args)
    
    ## Send data to manager
    def send_data():
        data_amt = 0
        for frame_data in dataloader:
            execution_manager.run(frame_data)
            #print ("Sending data to the pipeline #{0}".format(data_amt))
            data_amt += 1
            #if data_amt == 300:
            #    break
        execution_manager.run(None)

    t0 = time.time()
    Thread(target=send_data).start()
    
    ## Get results
    while True:
        result = execution_manager.get()
        if result == [None]:
            print ('Pipeline finished')
            break
        yield result
    
    print ("Total time spent on pipeline: {0}".format(time.time() - t0))
