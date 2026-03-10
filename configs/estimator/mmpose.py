model_args = dict(
    config = "libs/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py",
    checkpoint = "weights/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth",
)
ignored_classes = [2] # Classes that will not be tracked
