import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--detector', '-d', default='eva02', type=str, help='Detector architecture to use.')
    parser.add_argument('--tracker', '-t', default='smiletrack', type=str, help='Object tracker architecture to use.')
    parser.add_argument('--estimator', '-e', default='mmpose', type=str, help='Pose estimator architecture to use.')
    parser.add_argument('--causal-rules', '-c', default='base_rules', type=str, help='Causal rules to use.')
    #parser.add_argument('--dataloader', '-l', default='coco_loader', type=str, help='Dataloader to use.')
    parser.add_argument('--dataloader', '-l', default='video', type=str, help='Dataloader to use.')

    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use.')
    parser.add_argument('--threshold', default=0., type=float, help='Confirmed detection threshold.')
    
    ## Store-true arguments
    parser.add_argument('--return-intermediate', action='store_true', help='Return intermediate results.')

    # dataloader args
    #parser.add_argument("--data-path", type=str, default='samples/TestAnnots.json', help="Path to the dataset.")
    parser.add_argument("--data-path", type=str, default='samples/original.mp4', help="Path to the dataset.")
    parser.add_argument('--dets-path', default="samples/sample_results.json", type=str, help='Path to save the results.')

    return parser.parse_args()
