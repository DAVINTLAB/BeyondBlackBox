from templates.PoseEstimator import PoseEstimator

from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples

class mmpose(PoseEstimator):
    def __init__(self, args):
        super(mmpose, self).__init__(args)
        self.estimator_args = args.model_args
        self.expand_bbox = True
        self.expand_bbox_ratio = 0.25
    
    def __init_model__(self):
        self.pose_estimator = init_pose_estimator(
            cfg_options={"model": {"test_cfg": {"output_heatmaps": False}}},
            **self.estimator_args,
        )
            #args.pose_config,
            #args.pose_checkpoint,
            #device=args.device,

    def __pre_process__(self, image, detections):
        bboxes = [d['bbox'] for d in detections]
        if self.expand_bbox:
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                expand_x = (x2 - x1) * self.expand_bbox_ratio
                expand_y = (y2 - y1) * self.expand_bbox_ratio
                x1 = max(0, int(x1 - expand_x))
                x2 = min(image.shape[1], int(x2 + expand_x))
                y1 = max(0, int(y1 - expand_y))
                y2 = min(image.shape[0], int(y2 + expand_y))
                bboxes[i] = (x1, y1, x2, y2)
        return (image, detections, bboxes)

    def __estimate__(self, data):
        img, detections, bboxes = data

        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        pose_results = merge_data_samples(pose_results)
        pose_data = pose_results.get('pred_instances', None)

        if pose_data is None:
            return []

        assert len(pose_data) == len(bboxes), "Expected {} set of keypoints, got {}".format(len(bboxes), len(kps))
        for i in range(len(pose_data)):
            kp = pose_data[i].keypoints[0]
            score = pose_data[i].keypoint_scores[0]
            visibility = pose_data[i].keypoints_visible[0]

            detections[i]['pose'] = kp.tolist()
            detections[i]['pose_score'] = score.tolist()
            detections[i]['pose_visibility'] = visibility.tolist()

        return detections

    def __post_process__(self, result):
        return result

    def __apply_rules__(self, objs_with_skeleton, objs_without_skeleton, causal_rules):
        objs_with_skeleton = causal_rules.apply_rules_estimator(objs_with_skeleton, objs_without_skeleton)

        return objs_with_skeleton
