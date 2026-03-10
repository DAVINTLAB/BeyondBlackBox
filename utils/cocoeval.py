from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
import numpy as np
from tabulate import tabulate
import json
import sys
import io
import os

DETECTION_ACCUMULATOR = {"TP": 0, "FP": 0}

def compute_fdr(coco_eval, cat_id):
    false_positives = 0
    true_positives = 0

    for eval_img in coco_eval.evalImgs:
        if eval_img is None or eval_img['category_id'] != cat_id:
            continue

        dt_matches = eval_img['dtMatches'][0]  # Detection-to-GT matches at 50% IoU
        dt_ignore = eval_img['dtIgnore'][0]   # Flags indicating ignored detections at 50% IoU
        num_gt = np.sum(eval_img['gtIgnore'] == 0)  # Count valid ground truths

        # TP: Detections that match a GT and are not ignored
        tp = np.sum((dt_matches > 0) & (dt_ignore == 0))
        true_positives += tp

        # FP: Detections that do not match any GT and are not ignored
        fp = np.sum((dt_matches == 0) & (dt_ignore == 0))
        false_positives += fp

    if cat_id not in DETECTION_ACCUMULATOR:
        DETECTION_ACCUMULATOR[cat_id] = {"TP": 0, "FP": 0}
    DETECTION_ACCUMULATOR[cat_id]["TP"] += true_positives
    DETECTION_ACCUMULATOR[cat_id]["FP"] += false_positives
    DETECTION_ACCUMULATOR["TP"] += true_positives
    DETECTION_ACCUMULATOR["FP"] += false_positives

    # Calculate FDR for the current class
    total_predictions = true_positives + false_positives
    fdr = (false_positives / total_predictions) if total_predictions > 0 else float('nan')
    return fdr

def load_custom(det_path, custom_threshold):
    with open(det_path, 'r') as infile:
        data = json.load(infile)
    return [x for x in data if x['score'] >= custom_threshold]


def eval(det_path, gt_path='./samples/TestAnnots.json', custom_threshold=None):
    if custom_threshold is not None:
        det_path = load_custom(det_path, custom_threshold)

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        cocoGt = COCO(gt_path)
        cocoDt = cocoGt.loadRes(det_path)
   
        cocoEval = COCOeval(cocoGt,cocoDt, 'bbox')
    
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    finally:
        sys.stdout = original_stdout

    class_names = ["Armed", "Unarmed", "Gun"]
    metrics = {
            #"AP":0, 
            "AP50":1, 
            #"AP75":2,
            #"AR":8,
    }
    
    results = {
        metric: float(cocoEval.stats[idx] * 100 if cocoEval.stats[idx] >= 0 else "nan")
        for metric, idx in metrics.items()
    }
    
    precisions = cocoEval.eval["precision"]
    recalls = cocoEval.eval["recall"]
    scores = cocoEval.eval["scores"]
    results_per_category = []
    
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[0, :, idx, 0, -1] #### (IoU, dets, ...)
        recall = recalls[0, idx, 0, -1] #### (IoU, ...)
    
        precision = precision[precision > -1]
        recall = recall[recall > -1]
    
        ap = np.mean(precision) if precision.size else float("nan")
        ar = np.mean(recall) if recall.size else float("nan")
        fdr = compute_fdr(cocoEval, idx)
        prec = 1 - fdr
        f1 = 2 * (prec * ar) / (prec + ar) if prec + ar > 0 else float("nan")
    
        results_per_category.append(("{}".format(name), float(ap * 100), float(ar * 100), float(prec * 100), float(f1 * 100)))
    
    APs = [result[1] for result in results_per_category]
    ARs = [result[2] for result in results_per_category]
    PRECs = [result[3] for result in results_per_category]
    F1s = [result[4] for result in results_per_category]
    
    results_per_category.append(("Aggregated", np.mean(APs), np.mean(ARs), np.mean(PRECs), np.mean(F1s)))
    results["PREC"] = np.mean(PRECs)
    results["AR"] = np.mean(ARs)
    results["F1"] = np.mean(F1s)
    return results, results_per_category

def print_categories(results_per_category):
    # tabulate class results
    N_COLS = 5
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    class_table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP", "AR", "PREC", "F1"],
        numalign="left",
    )
    print ("Per-category bbox metrics: \n" + class_table)
    print ('')

def print_results(results):
    # tabulate bbox metrics
    N_COLS = len(results[0])
    results_flatten = list(itertools.chain(*[list(r.values()) for r in results]))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    metrics_table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=list(results[0].keys()),
        numalign="left",
    )
    print ("Performance metrics: \n" + metrics_table)



experiment_results = [
    #'./samples/EVA_test_seeded.json',
    #'./samples/EVA_std_seeded.json',
#        './samples/stage0_seed0conf_testset.json',
#    './samples/stage0_0nms_testset.json',
#        './samples/stage2_0nms_testset.json',
#    './samples/stage0_8nms_testset.json',
#    './samples/stage1_8nms_testset.json',
#    './samples/stage2_8nms_testset.json',
#    './samples/stage1_8nms_5s_testset.json',
#    './samples/stage2_8nms_5s_testset.json',
#    './samples/stage0_mmpose_vit_testset.json',
#    './samples/EVA_filtered.json',
#    './samples/stage1_mmpose_vit_testset.json',
#    './samples/stage2_mmpose_vit_testset.json',
#    './samples/stage2_mmpose_vit_box_estimation_testset.json',
#    './samples/filtered.json',
#    './samples/stage2_mmpose_vit_box_estimation_kh_testset.json',
#    './samples/filtered_kh.json',
#    './samples/filtered_kh_custom.json',
#    './samples/stage2_reversed_3fcap_testset.json',
#    './samples/stage2_reversed_new_checks_testset.json',
#    './samples/stage2_reversed_no_estimation_testset.json',
     #'./samples/filtered_reversed.json',
#    './samples/stage0_reversed_skip_testset.json',
#    './samples/stage0_filtered_skip.json',
#    './samples/stage2_spedup_testset.json',
#    './samples/stage2_filtered_spedup_testset.json',
#    './samples/stage2_3xspedup_testset.json',
#    './samples/stage2_filtered_3xspedup_testset.json',
     #"./samples/latest.json",
#    "./samples/latest_filtered.json",
#    "./samples/stage2_skip1_0.json",
    #"./samples/stage2_baseNoEst_0.json",
    #"./samples/stage2_baseNoEst_0_filtered.json",
    #"./samples/stage0_0_yolo.json",
    #"./samples/stage2_0_yolo.json",
    #"./samples/filteredstage0_0_yolo.json",
    #"./samples/filteredstage2_0_yolo.json",
    #"./samples/stage0_0_yolo_640.json",
    #"./samples/stage2_0_yolo_640.json",
    #"./samples/filteredstage0_0_yolo_640.json",
    #"./samples/filteredstage2_0_yolo_640.json",
    #"./samples/stage0_0_yolo_11_1024.json",
    #"./samples/stage2_0_yolo_11_1024.json",
    #"./samples/filteredstage0_0_yolo_11_1024.json",
    #"./samples/filteredstage2_0_yolo_11_1024.json",
    #"./samples/stage2_0_yolo_11_est_1024.json",
    #"./samples/stage2_ablation#2.json", ## Temporal (H3 + H2)
    #"./samples/stage2_ablation#3.json", ## Temporal + Ownership (H3 + H2 + H1)
    #"./samples/stage2_ablation#4.json", ## Temporal + Insertion (H3 + H2 + H4) 
    #"./samples/stage2_ablation#5.json", ## Ownership (H1)
    #"./samples/stage2_ablation#6.json", ## Inerstion (H4)

     #"./samples/new_heuristic_results.json",
     ##"./samples/new_heuristic_voting_results.json",
     #"./samples/new_heuristic_new_voting_results.json",
     #"./samples/new_heuristic_best_voting_results.json",
     #"./samples/new_heuristic_newest_voting_results.json",
     #
     #"./samples/newest_heuristic_results.json",
     ##"./samples/newest_heuristic_voting_results.json",
     #"./samples/newest_heuristic_new_voting_results.json",
     #"./samples/newest_heuristic_best_voting_results.json",
     ##"./samples/newest_heuristic_newest_voting_results.json",
     #
     #"./samples/voting_results.json",

    #"./samples/wi2fps1_base_results.json",
    #"./samples/wi2fps1_results.json",
    #"./samples/wi2fps1_baseline_results.json",
    #"./samples/wi2fps1_cascading_results.json",
    #"./samples/wi2fps1_voting_results.json",
    #"./samples/wi2fps5_base_results.json",
    #"./samples/wi2fps5_results.json",
    #"./samples/wi2fps7_base_results.json",
    #"./samples/wi2fps7_results.json",

    #"./samples/gdd_base_results.json",
    "./samples/gdd_baseline_results.json",
    "./samples/gdd_cascading_results.json",
    "./samples/gdd_voting_results.json",
]

#for stage_id in [0,2]:
#    for base_id in range(10):
#        experiment_results.append(f'samples/stage{stage_id}_base_{base_id}.json')

#experiment_results = [experiment_results[0], experiment_results[-1]]

#GT_PATH='./samples/TestAnnots.json'
#GT_PATH='/home/liswsz6-02/doc/weapons_images_2fps/Cam1_coco_annotations.json'
#GT_PATH='/home/liswsz6-02/doc/weapons_images_2fps/Cam5_coco_annotations.json'
#GT_PATH='/home/liswsz6-02/doc/weapons_images_2fps/Cam7_coco_annotations.json'
GT_PATH='/home/liswsz6-02/doc/YouTube-GDD/coco_annotations_test.json'

def run_evals():
    metrics_achieved = []
    for det_path in experiment_results:
        for k in ["TP", "FP"]:
            DETECTION_ACCUMULATOR[k] = 0
        results, results_per_category = eval(det_path, gt_path=GT_PATH)
    
        experiment_name = os.path.basename(os.path.splitext(det_path)[0])
        results["Experiment"] = experiment_name
    
        print ("Results for experiment: " + experiment_name)
        
        #print ("Detections were divided between: {}FPs and {}TPs".format(DETECTION_ACCUMULATOR["FP"], DETECTION_ACCUMULATOR["TP"]))
        #print ("Per-class divisions:")
        #for cls in range(3):
        #    print ("Class {}: {}FPs and {}TPs".format(cls, DETECTION_ACCUMULATOR[cls]["FP"], DETECTION_ACCUMULATOR[cls]["TP"]))
    
        print_categories(results_per_category)
        results = dict(sorted(results.items(), key=lambda x: -len(x[0])))
        metrics_achieved.append(results)
    
    print_results(metrics_achieved)


def run_curve(metric_name="FDR"):
    all_results = {}
    for det_path in experiment_results:
        exp = []
        for i in range(10):
            threshold = i / 10
            results, results_per_category = eval(det_path, custom_threshold=threshold)
            experiment_name = os.path.basename(os.path.splitext(det_path)[0])

            metric = results[metric_name]
            # f1s = [result[4] for result in results_per_category]
            exp.append(metric)
        all_results[experiment_name] = exp
    # print (all_results)
    print (','.join(list(all_results)))
    for i in range(10):
        line = []
        for k, v in all_results.items():
            line.append(str(v[i]))
        print (','.join(line))


run_evals()
#run_curve()
