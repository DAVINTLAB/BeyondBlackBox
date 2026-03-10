from glob import glob
import json
import os
try:
    from utils.to_coco_results import merge_files, read_dataset_file, read_vid_dims
except ImportError:
    from to_coco_results import merge_files, read_dataset_file, read_vid_dims

def voting(obj, THRESHOLD=6):
    score = obj['score'] * 10
    h_results = obj.get("h-results", {})

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


def preprocess_dataset(dataset_file):
    with open(dataset_file) as infile:
        data = json.load(infile)

    dataset = {}
    for obj in data['annotations']:
        img_id = obj['image_id']
        if img_id not in dataset:
            dataset[img_id] = []
        dataset[img_id].append(obj)
    return dataset

def remap_json(json_obj):
    mapped = {}
    for item in json_obj:
        img_id = item['image_id']
        if img_id not in mapped:
            mapped[img_id] = []
        mapped[img_id].append({
            'bbox': item['bbox'],
            'category_id': item['category_id'],
            'image_id': item['image_id'],
            'score': item['score'],
            'h-results': item.get('h-results', {})
        })
    return mapped


def IoU(bbx1, bbx2):
    x1 = max(bbx1[0], bbx2[0])
    y1 = max(bbx1[1], bbx2[1])
    x2 = min(bbx1[0] + bbx1[2], bbx2[0] + bbx2[2])
    y2 = min(bbx1[1] + bbx1[3], bbx2[1] + bbx2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bbx1_area = bbx1[2] * bbx1[3]
    bbx2_area = bbx2[2] * bbx2[3]

    iou = inter_area / float(bbx1_area + bbx2_area - inter_area)
    return iou


def find_matching_bbox(bbx, candidates, iou_threshold=0.5):
    best_candidate = None
    best_iou = 0.0
    for candidate in candidates:
        iou = IoU(bbx, candidate['bbox'])
        if iou >= iou_threshold:
            if iou > best_iou:
                best_iou = iou
                best_candidate = candidate
    return best_candidate


def assign_gt(detections, gt_data, iou_threshold=0.5):
    assigned = []
    matched = set()
    for det in detections:
        match = find_matching_bbox(det['bbox'], gt_data, iou_threshold)
        if match:
            assigned.append((det, match))
            matched.add(match['id'])
        else:
            assigned.append((det, None))

    for gt in gt_data:
        if gt['id'] not in matched:
            assigned.append((None, gt))
    return assigned


def make_output(assignments):
    output_data = []
    id_counter = 0
    for det, gt in assignments:
        output = {
            "image_id": None,
            "id": id_counter,
            "IoU threshold": 0.5,
            "predicted label": None,
            "ground truth label": None,
            "prediction score": None,
            "heuristic results": None,
            "decision": None,
        }

        if det:
            output['image_id'] = det.get('image_id', None)
            output['predicted label'] = det.get('category_id', None)
            output['prediction score'] = det.get('score', None)
            output['heuristic results'] = det.get('h-results', None)
            # heuristics_passed = [v['satisfied'] for v in output['heuristic results'].values()]
            voting_result = voting(det)
            heuristics_passed = [voting_result]

            if all(heuristics_passed):
                output['decision'] = "Prediction accepted"
            else:
                output['decision'] = "Prediction discarded"

        else:
            output['decision'] = "Failed to detect object"

        if gt:
            output['image_id'] = gt.get('image_id', None)
            output['ground truth label'] = gt.get('category_id', None)
        
        
        output_data.append(output)
        id_counter += 1

    return output_data


def main(folder, dataset_file, vid_info, iou_threshold=0.5):
    files = glob(os.path.join(folder, '*.json'))
    relevant_frames = read_dataset_file(dataset_file)
    vid_dims = read_vid_dims(vid_info)
    merged = merge_files(files, relevant_frames, vid_dims, keep_frame_ids, keys_to_keep=["h-results", "score"])

    remapped_detections = remap_json(merged)
    dataset = preprocess_dataset(dataset_file)

    final_output = []
    for img_id, dets in remapped_detections.items():
        gt_data = dataset.get(img_id, [])
        assignments = assign_gt(dets, gt_data, iou_threshold)
        output_data = make_output(assignments)
        final_output.extend(output_data)

    return final_output

if __name__ == '__main__':
    ## Dets folder
    dets_src = 'samples/testset/results/newest_heuristic_results/'
    ## GT file
    dataset = 'samples/TestAnnots.json'
    ## Info file
    vid_info = 'samples/vid_info.json'
    ## Save file
    save_file = f'samples/newest_pipeline_heuristics.json'
    ## Are the frame_ids correct already?
    keep_frame_ids = False

    output = main(dets_src, dataset, vid_info, iou_threshold=0.5)

    with open(save_file, 'w') as outfile:
        json.dump(output, outfile)
    print(f'Saved {len(output)} objects to {save_file}')
