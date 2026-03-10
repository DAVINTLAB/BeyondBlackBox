import os
import utils
import json

def main(args):
    detections = []
    for results in utils.runner.run(args):
        if not len(detections):
            for _ in range(len(results)):
                detections.append([])

        for idx, result in enumerate(results):
            for obj in result:
                detections[idx].append(obj)
    
    for i, det in enumerate(detections):
        path, name = os.path.split(args.dets_path)
        if args.return_intermediate:
            save_name = f"stage#{i}_{name}"
        else:
            save_name = name

        with open(os.path.join(path, save_name), "w") as outfile:
            json.dump(det, outfile)

        print ("Data written to: ", os.path.join(path, save_name))

if __name__ == '__main__':
    args = utils.argparser.parse_args()
    main(args)
    print ("Main finished")
