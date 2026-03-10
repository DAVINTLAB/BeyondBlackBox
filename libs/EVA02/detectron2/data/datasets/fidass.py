from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'name': 'Armed', 'id': 0},
    {'name': 'Unrmed', 'id': 1},
    {'name': 'Gun', 'id': 2},
]

#categories = [
#    {'name': 'Person', 'id': 1},
#    {'name': 'Gun', 'id': 2},
#]
 

splits = {
	"fidass_train" : ('coco', os.path.join('coco', 'annotations', 'TrainAnnots.json')),
	"fidass_val" : ('coco', os.path.join('coco', 'annotations', 'ValidationAnnots.json')),
	"fidass_test" : ('coco', os.path.join('coco', 'annotations', 'TestAnnots.json')),
}

for key, (image_root, json_file) in splits.items():
	register_coco_instances(
		key,
		{},
		os.path.join("datasets", json_file) if "://" not in json_file else json_file,
		os.path.join("datasets", image_root),
	)
