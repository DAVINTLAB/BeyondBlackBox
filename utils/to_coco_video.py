import json
import os
from glob import glob
import unidecode

def merge_files(files, relevant_frames, vid_dims, keep_frame_ids):
    merged = {}
    for file in files:
        with open(file) as infile:
            data = json.load(infile)

        if not keep_frame_ids:
            vidname = os.path.basename(file).split('.')[0]

            if vidname.startswith('stage#'):
                vidname = '_'.join(vidname.split('_')[1:])

            standardized_name = vidname.lower().replace(' ', '_')
            standardized_name = unidecode.unidecode(standardized_name)
            assert standardized_name in relevant_frames, f'{standardized_name} not in annotation set'

        if type(data[0]) == list:
            data = sum(data, [])

        merged[standardized_name] = []

        for obj in data:
            frame = obj['fid']
            if not keep_frame_ids:
                if frame not in relevant_frames[standardized_name]:
                    continue

            score = obj['score']
            cls = obj['class']
            bbox = obj['bbox']

            ## convert bbx from tlbr to tlwh
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            if not keep_frame_ids and False:
                ## scale bbx from original size to 768x480
                original_dims = {"width":1024,"height":1024} # vid_dims[vidname]
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
                'category_id': cls,
                'bbox': bbox,
                'score': score,
            }

            merged[standardized_name].append(std_obj)

    return merged

def read_dataset_file(dataset):
    with open(dataset) as infile:
        data = json.load(infile)

    relevant_frames = {}
    vid_annots = {}
    for obj in data['images']:
        file = obj["file_name"]
        vidname = file.split('/')[-2]

        if vidname not in relevant_frames:
            relevant_frames[vidname] = {}

        frame = file.split('/')[-1].split('.')[0]
        frame = int(frame.split('_')[-1])

        relevant_frames[vidname][frame] = obj['id']
        if vidname not in vid_annots:
            vid_annots[vidname] = []
        vid_annots[vidname].append(obj)

    return vid_annots, relevant_frames

def read_vid_dims(vid_info):
    with open(vid_info) as infile:
        return json.load(infile)


if __name__ == '__main__':
    for stage_id in [2]:
        for base_id in range(1):
            ## Dets folder
            dets_src = f'samples/testset/stage_{stage_id}_baseNoEst_{base_id}'
            ## GT file
            dataset = 'samples/TestAnnots.json'
            ## Info file
            vid_info = 'samples/vid_info.json'
            ## Save file
            save_file = f'samples/vids_stage{stage_id}_baseNoEst_{base_id}.json'
            ## Are the frame_ids correct already?
            keep_frame_ids = False
            
            ### Dets folder
            #dets_src = 'samples/testset/stage_2_base_0'
            ### GT file
            #dataset = 'samples/TestAnnots.json'
            ### Info file
            #vid_info = 'samples/vid_info.json'
            ## Save file
            #save_file = 'samples/stage2_3xspedup_testset.json'
            ## Are the frame_ids correct already?
            #keep_frame_ids = False
        
        
            vid_annots, relevant_frames = read_dataset_file(dataset)
        
            vid_dims = {}#read_vid_dims(vid_info)
            if keep_frame_ids:
                files = [dets_src]
            else:
                files = glob(os.path.join(dets_src, '*.json'))
        
            merged = merge_files(files, relevant_frames, vid_dims, keep_frame_ids)
        
            print ("Saving merged annotations to", save_file)
            with open(save_file, 'w') as outfile:
                json.dump(merged, outfile)

            with open("samples/vid_gt.json", 'w') as outfile:
                json.dump(vid_annots, outfile)
