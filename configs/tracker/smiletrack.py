video_framerate = 30 # Framerate of the video
ignored_classes = [2] # Classes that will not be tracked
track_high_thresh = 0.3 # tracking confidence threshold
track_low_thresh = 0.1 # lowest detection threshold
new_track_thresh = 0.33 # new track thresh
min_box_area = 40 # minimum box area
track_buffer = 105 # window size for tracking
match_thresh = .9 # matching threshold for tracking
min_appearance = 30 # minimum frames for each track

##ReID parameters
mot20 = False # fuse score and iou for association
cmc_method = "sparseOptFlow" # cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc
with_reid = False # with ReID module.
fast_reid_weights = "pretrained/mot17_sbs_S50.pth" # reid weights file path
proximity_thresh = 0.5 # threshold for rejecting low overlap reid matches
appearance_thresh = 0.25 # threshold for rejecting low appearance similarity reid matches
