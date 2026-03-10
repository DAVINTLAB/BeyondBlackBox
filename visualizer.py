import cv2
import os
import numpy as np

CLASSES = [
    "Armed",
    "Unarmed",
    "Gun"
]

## Class for visualizing the output predictions
## Receives a dataloader and a list of predictions
## Iterates over each data point and returns it with the predictions drawn over
class Visualizer:
    def __init__(self, dataloader, predictions, pose_plotter=None, save_images=False):
        self.dataloader = dataloader
        self.predictions = self.__prepare_preds__(predictions)
        self.pose_plotter = pose_plotter
        self.fid = 0
        self.iterator = None
        self.save_images = save_images

    def __prepare_preds__(self, preds):
        predictions = {}
        for pred in preds:
            if pred['fid'] not in predictions:
                predictions[pred['fid']] = []
            predictions[pred['fid']].append(pred)
        return predictions

    def __iter__(self):
        self.fid = 0
        self.iterator = self.dataloader.__iter__()
        for _ in range(self.fid):
            next(self.iterator)
        self.fid = 1
        return self

    def __next__(self):
        frame = next(self.iterator)['image']

        if frame is None:
            raise StopIteration
    
        if self.fid not in self.predictions:
            self.fid += 1
            return frame

        for pred in self.predictions[self.fid]:
            bbox = pred.get('bbox', None)
            score = pred.get('score', None)
            lbl = pred.get('class', None)
            tid = pred.get('tid', None)
            
            if score is None or score < 0.5:
                continue

            x1, y1, x2, y2 = bbox
            if self.save_images:
                rect_color = (255, 255, 0)
                cv2.putText(frame, CLASSES[lbl], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2)
            else:
                rect_color = (0, 255, 0)
                cv2.putText(frame, str(int(lbl)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2)
                #cv2.putText(frame, str(tid), (x1+30, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2)
                #cv2.putText(frame, str(round(score,2)), (x1+60, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
            #if 'match' in pred and not self.save_images:
            #    if pred['match']:
            #        cv2.circle(frame, (x1,y2), 5, (0, 0, 255), -1)
            #        if 'infered' in pred:
            #            cv2.circle(frame, (x2,y1), 5, (255, 0, 0), -1)

            if 'attn_map' in pred:
                attn_map = pred['attn_map']
                attn_map = np.asarray(attn_map).reshape(64,64)
                attn_map = cv2.resize(attn_map, (768, 480), interpolation=cv2.INTER_LINEAR)
                attn_map = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)
                frame = cv2.addWeighted(frame, 1, attn_map, .5, 0)

            if self.pose_plotter is not None and len(pred['pose']):
                relevant = pred['pose_score'][:11]
                head = pred['pose_score'][:5]
                left_arm = pred['pose_score'][5:8]
                right_arm = pred['pose_score'][8:11]

                valid_pose = any([score > 0.5 for score in head]) and (any([score > 0.5 for score in left_arm]) or any([score > 0.5 for score in right_arm]))

                if valid_pose:
                    frame = self.pose_plotter(frame, np.asarray([pred['pose']]))
                    #cv2.circle(frame, (int(pred['pose'][1][0]), int(pred['pose'][1][1])), 5, (0, 255, 255), -1)
                    #cv2.circle(frame, (int(pred['pose'][9][0]), int(pred['pose'][9][1])), 5, (255, 0, 255), -1)
                    #cv2.circle(frame, (int(pred['pose'][10][0]), int(pred['pose'][10][1])), 5, (255, 0, 255), -1)
                    #cv2.putText(frame, str(round(pred['pose_score'][9],2)), (int(pred['pose'][9][0]), int(pred['pose'][9][1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
                    #cv2.putText(frame, str(round(pred['pose_score'][10],2)), (int(pred['pose'][10][0]), int(pred['pose'][10][1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
                    #cv2.putText(frame, str(round(pred['pose_visibility'][9],2)), (int(pred['pose'][9][0]), int(pred['pose'][9][1])+30), cv2.FONT_HERSHEY_SIMPLEX, .5, (127, 0, 127), 1)
                    #cv2.putText(frame, str(round(pred['pose_visibility'][10],2)), (int(pred['pose'][10][0]), int(pred['pose'][10][1])+30), cv2.FONT_HERSHEY_SIMPLEX, .5, (127, 0, 127), 1)

        self.fid += 1
        return frame


if __name__ == '__main__':
    import argparse
    import implementations
    import json

    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--dataloader', '-l', default='video', type=str, help='Dataloader to use.')
    parser.add_argument("--data-path", type=str, default='samples/original.mp4', help="Path to the dataset.")
    parser.add_argument("--data-prefix", type=str, default='.', help="Prefix to the the dataset's images.")
    parser.add_argument("--predictions", '-p', type=str, default='samples/sample_results.json', help="Path to the predictions file.")
    parser.add_argument("--draw-poses", action='store_true', help="Draw poses on the output.")
    parser.add_argument("--save-images", action='store_true', help="Save video instead of displaying.")
    args = parser.parse_args()

    dataloader = implementations.load('dataloader', args.dataloader, args)
    with open(args.predictions, 'r') as infile:
        predictions = json.load(infile)

    if args.draw_poses:
        from libs.estimator.utils.vis import plot_poses
    else:
        plot_poses = None

    visualizer = Visualizer(dataloader, predictions, plot_poses, save_images=args.save_images)
    video_recorder = None

    for frame in visualizer:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if args.save_images:
            if video_recorder is None:
                det_path = os.path.splitext(args.predictions)[0]
                video_recorder = cv2.VideoWriter(f'{det_path}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
            video_recorder.write(frame)
        else:
            cv2.imshow('frame', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if video_recorder is None:
        cv2.destroyAllWindows()
    else:
        video_recorder.release()
        cv2.destroyAllWindows()
        print (f"Video saved to {det_path}.avi")

