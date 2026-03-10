import os
from glob import glob

num_iters = 1
stage_id = 2
base_folder_name = "stage_{}_baseNoEst_{}"
skip = ""
vid_root = "./samples/testset/"

files = glob(os.path.join(vid_root, "*.mp4"))

for i in range(num_iters):
    target_folder = os.path.join(vid_root, base_folder_name.format(stage_id,i))
    print (target_folder)

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for file in files:
        base_name = os.path.basename(file).split('.')[0]
        json_orig = f"stage#{stage_id}_{base_name}{skip}{i}.json"
        json_targ = f"{base_name}.json"

        path_json_orig = os.path.join(vid_root, json_orig)
        path_json_targ = os.path.join(vid_root, base_folder_name.format(stage_id,i), json_targ)
        #jsonpath = file.replace('.mp4', f'{i}.json')
        #base_name = os.path.basename(file).split('.')[0] + '.json'
        print(path_json_orig, path_json_targ)
        os.rename(path_json_orig, path_json_targ)
