import json
import os
from glob import glob
import unidecode



## Optimal1 : [6, 0,2, 1,3, 8,2, 1,10, 1,0, 0,10]
## Optimal2 : [6, 0,2, 0,0, 7,1, 1,10, 1,0, 0,10]
def voting2(obj, THRESHOLD=6):
    score = obj['score'] * 10
    h_results = obj.get("h-results", {})
    print(h_results)

    if 'matching skeleton' in h_results:
        if h_results['matching skeleton']['satisfied']: score += 0
        else: score -= 2

        if h_results['best match']['satisfied']: score += 1
        else: score -= 3

    else:
        if h_results['valid pose']['satisfied']: score += 8
        else: score -= 2

        if h_results['short track']['satisfied']: score += 1
        else: score -= 10

        if h_results['consistent pose']['satisfied']: score += 1
        else: score -= 0

        if h_results['reliable track']['satisfied']: score += 0
        else: score -= 10

    return score > THRESHOLD

def voting1(obj, THRESHOLD=6):
    score = obj['score'] * 5
    h_results = obj.get("h-results", {})

    if 'matching skeleton' in h_results:
        if h_results['matching skeleton']: score += 2

        if h_results['best match']: score += 2

    else:
        if h_results['valid pose']: score += 2

        if 1-h_results['noised track']: score += 2

        if h_results['consistent pose']: score += 2

        if h_results['reliable track']: score += 2

    return score >= THRESHOLD

def merge_files(files, relevant_frames, vid_dims, keep_frame_ids, keys_to_keep=[]):
    merged = []
    for file in files:
        with open(file) as infile:
            data = json.load(infile)

        if not keep_frame_ids:
            vidname = os.path.basename(file).split('.')[0]

            if vidname.startswith('stage#'):
                vidname = '_'.join(vidname.split('_')[1:])

            if "base" in file:
                vidname = vidname[:-1]

            standardized_name = vidname.lower().replace(' ', '_')
            standardized_name = unidecode.unidecode(standardized_name)
            assert standardized_name in relevant_frames, f'{standardized_name} not in annotation set'

        if not len(data):
            print(file)
            continue

        if type(data[0]) == list:
            data = sum(data, [])

        for obj in data:
            frame = obj['fid']
            if not keep_frame_ids:
                if frame not in relevant_frames[standardized_name]:
                    continue

            score = obj['score']
            cls = obj['class']
            bbox = obj['bbox']

            if "h-results" in obj:# and False:
                #passes = all([v['satisfied'] for v in obj["h-results"].values()]) or score > .9
                #if not passes:
                if not voting2(obj):
                    continue

            ## convert bbx from tlbr to tlwh
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            if keep_frame_ids and False:
            #if keep_frame_ids and False:
                ## scale bbx from original size to 768x480
                # original_dims = {"width":1024,"height":1024} # vid_dims[vidname]
                original_dims = vid_dims
                scale_x = 768 / original_dims['width']
                scale_y = 480 / original_dims['height']

                bbox[0] *= scale_x
                bbox[2] *= scale_x
                bbox[1] *= scale_y
                bbox[3] *= scale_y

            bbox = list(map(lambda x: int(round(x)), bbox))

            if keep_frame_ids:
                fid = frame
            else:
                fid = relevant_frames[standardized_name][frame]

            std_obj = {
                'image_id': fid,
                'category_id': cls if cls == 2 else 0,
                'bbox': bbox,
                'score': score,
            }

            for key in keys_to_keep:
                if key in obj:
                    std_obj[key] = obj[key]

            merged.append(std_obj)

    return merged

def read_dataset_file(dataset):
    with open(dataset) as infile:
        data = json.load(infile)

    relevant_frames = {}
    for obj in data['images']:
        file = obj["file_name"]
        vidname = file.split('/')[-2]

        if vidname not in relevant_frames:
            relevant_frames[vidname] = {}

        frame = file.split('/')[-1].split('.')[0]
        frame = int(frame.split('_')[-1])

        relevant_frames[vidname][frame] = obj['id']

    return relevant_frames

def read_vid_dims(vid_info):
    if not vid_info:
        return {"height": 720, "width": 1280}

    with open(vid_info) as infile:
        return json.load(infile)


def map_detections_do_dataset(dataset, dets_src, vid_info, keep_frame_ids, keys_to_keep=[]):
    if not keep_frame_ids:
        relevant_frames = read_dataset_file(dataset)
    else:
        relevant_frames = None
        
    vid_dims = read_vid_dims(vid_info)
    if keep_frame_ids:
        files = [dets_src]
    else:
        files = glob(os.path.join(dets_src, '*.json'))
        
    merged = merge_files(files, relevant_frames, vid_dims, keep_frame_ids, keys_to_keep)
    return merged



if __name__ == '__main__' and False:
    ## Dets folder
    # dets_src = f'samples/testset/stage_{stage_id}_baseNoEst_{base_id}'
    # dets_src = f'samples/testset/yolo_1024_est_stage_{stage_id}_base_{base_id}'
    dets_src = "samples/testset/results/new_heuristic_results/"
    #dets_src = "samples/testset/results/heuristic_results/"
    ## GT file
    dataset = 'samples/TestAnnots.json'
    ## Info file
    vid_info = 'samples/vid_info.json'
    ## Save file
    save_file = f'samples/voting_results.json'
    #save_file = f'samples/newest_heuristic_best_voting_results.json'
    ## Are the frame_ids correct already?
    keep_frame_ids = False

    merged = map_detections_do_dataset(dataset, dets_src, vid_info, keep_frame_ids)

    with open(save_file, 'w') as outfile:
        json.dump(merged, outfile)
    print(f'Saved {len(merged)} objects to {save_file}')
    
if __name__ == '__main__' and False:
    path = "~/doc/AblationTool/ablation_data/fidass/"

    for folder in glob(os.path.expanduser(path)+"*"):
        print(folder)
        if "stage" in folder:
            identifier = folder.split("/")[-1].split("_")[1]
            ## Dets folder
            dets_src = folder
            ## GT file
            dataset = 'samples/TestAnnots.json'
            ## Info file
            vid_info = 'samples/vid_info.json'
            ## Save file
            save_file = f'samples/stage2_{identifier}.json'
            ## Are the frame_ids correct already?
            keep_frame_ids = False
            merged = map_detections_do_dataset(dataset, dets_src, vid_info, keep_frame_ids)
        
            with open(save_file, 'w') as outfile:
                json.dump(merged, outfile)
            print(f'Saved {len(merged)} objects to {save_file}')
                

if __name__ == '__main__' and False:
    ## Dets folder
    dets_src = "samples/wi2fps1_newer.json"
    ## GT file
    dataset = '/home/liswsz6-02/doc/weapons_images_2fps/Cam1_coco_annotations.json'
    ## Info file
    vid_info = 'samples/vid_info.json'
    ## Save file
    save_file = f'samples/wi2fps1_nvoting_results.json'
    ## Are the frame_ids correct already?
    keep_frame_ids = True

    merged = map_detections_do_dataset(dataset, dets_src, vid_info, keep_frame_ids)

    with open(save_file, 'w') as outfile:
        json.dump(merged, outfile)
    print(f'Saved {len(merged)} objects to {save_file}')

if __name__ == '__main__' and True:
    ## Dets folder
    dets_src = "samples/gdd.json"
    ## GT file
    dataset = '/home/liswsz6-02/doc/YouTube-GDD/coco_annotations_test.json'
    ## Info file
    vid_info = 'samples/vid_info.json'
    ## Save file
    save_file = f'samples/gdd_voting_results.json'
    ## Are the frame_ids correct already?
    keep_frame_ids = True

    merged = map_detections_do_dataset(dataset, dets_src, vid_info, keep_frame_ids)

    with open(save_file, 'w') as outfile:
        json.dump(merged, outfile)
    print(f'Saved {len(merged)} objects to {save_file}')
