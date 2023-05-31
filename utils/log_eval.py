import os
import json
import torch
import numpy as np

from typing import List
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.util import read_txt


class EvalLog:
    def __init__(self, class_names: List, num_classes: int = 80, log_dir: str = 'logs'):
        self.class_names, self.num_classes, self.log_dir = class_names, num_classes, log_dir
        self.coco_maps, self.epochs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [0]
        if not os.path.exists(os.path.join(self.log_dir, "ground-truth/")):
            os.makedirs(os.path.join(self.log_dir, "ground-truth/"))
        if not os.path.exists(os.path.join(self.log_dir, "detection-results/")):
            os.makedirs(os.path.join(self.log_dir, "detection-results/"))
        if not os.path.exists(os.path.join(self.log_dir, 'coco_eval/')):
            os.makedirs(os.path.join(self.log_dir, 'coco_eval/'))

    def log_eval_txt(
            self, pred: torch.Tensor, gt: torch.Tensor, step: int, batch_size: int = 4
    ):
        assert len(pred) == gt.shape[0], '预测和标签的 batch_size 不匹配'
        for i, boxes in enumerate(pred):
            txt_name = '%06d.txt' % (step * batch_size + i)
            with open(os.path.join(self.log_dir, "detection-results/" + txt_name), mode="w", encoding='UTF-8') as f:
                for box in boxes:
                    x1, y1, x2, y2, class_conf, class_id = box
                    class_name = self.class_names[int(class_id)]
                    f.write("%s %s %s %s %s %s\n" % (
                        class_name, str(float(class_conf)),
                        str(float(x1)), str(float(y1)), str(float(x2)), str(float(y2))
                    ))
        for i, boxes in enumerate(gt):
            txt_name = '%06d.txt' % (step * batch_size + i)
            with open(os.path.join(self.log_dir, "ground-truth/" + txt_name), mode="w", encoding='UTF-8') as f:
                for box in boxes:
                    class_id, x1, y1, x2, y2 = box
                    if x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0:
                        break
                    class_name = self.class_names[int(class_id)]
                    f.write("%s %s %s %s %s\n" % (
                        class_name, str(float(x1)), str(float(y1)), str(float(x2)), str(float(y2))
                    ))

    def log_eval_map(self, epoch: int):
        print('=' * 100)
        ground_truth_path = os.path.join(self.log_dir, 'ground-truth')
        detection_path = os.path.join(self.log_dir, 'detection-results')
        coco_path = os.path.join(self.log_dir, 'coco_eval')
        ground_truth_json_path = os.path.join(coco_path, 'instances_ground_truth.json')
        detection_json_path = os.path.join(coco_path, 'instances_detection.json')
        with open(ground_truth_json_path, "w") as f:
            results_gt = self.preprocess_ground_truth(ground_truth_path=ground_truth_path)
            json.dump(results_gt, f, indent=4)
        with open(detection_json_path, "w") as f:
            results_dr = self.preprocess_detection(detection_path=detection_path)
            json.dump(results_dr, f, indent=4)
            if len(results_dr) == 0:
                print("未检测到任何目标。")
                print('=' * 100)
                return
        coco_ground_truth = COCO(ground_truth_json_path)
        coco_detection = coco_ground_truth.loadRes(detection_json_path)
        coco_eval = COCOeval(coco_ground_truth, coco_detection, 'bbox')
        coco_eval.evaluate()  # run per image evaluation（评估每个图像）
        coco_eval.accumulate()  # accumulate per image results（累积每个图像的结果）
        coco_eval.summarize()  # display summary metrics of results（显示结果的总结指标）
        coco_map = coco_eval.stats  # result summarization（结果总结）
        self.coco_maps.append(coco_map)
        self.epochs.append(epoch)
        with open(os.path.join(self.log_dir, 'coco_map.txt'), mode='a', encoding='UTF-8') as f:
            f.write(str(coco_map) + '\n')
        print('=' * 100)
        self.plot_map()

    def plot_map(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.epochs, np.array(self.coco_maps)[:, 0], 'red', linewidth=2, label='Ap@0.5:0.95')
        plt.plot(self.epochs, np.array(self.coco_maps)[:, 1], 'yellow', linewidth=2, label='Ap@0.5')
        plt.plot(self.epochs, np.array(self.coco_maps)[:, 3], 'black', linewidth=2, label='Ap@small')
        plt.plot(self.epochs, np.array(self.coco_maps)[:, 4], 'green', linewidth=2, label='Ap@medium')
        plt.plot(self.epochs, np.array(self.coco_maps)[:, 5], 'blue', linewidth=2, label='Ap@large')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('map')
        plt.title('map curve')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.log_dir, 'coco_map.png'))
        plt.cla()
        plt.close('all')

    def preprocess_ground_truth(self, ground_truth_path):
        image_ids = os.listdir(ground_truth_path)
        images = []
        bboxes = []
        for i, image_id in enumerate(image_ids):
            lines_list = read_txt(path=os.path.join(ground_truth_path, image_id))
            image_id = os.path.splitext(image_id)[0]
            image = {
                'file_name': image_id + '.jpg',
                'width': 1,
                'height': 1,
                'id': str(image_id)
            }
            boxes_per_image = []
            for line in lines_list:
                if line != '':
                    difficult = 0
                    if "difficult" in line:
                        line_split = line.split()
                        x1, y1, x2, y2, _difficult = line_split[-5:]
                        class_name = ""
                        for name in line_split[:-5]:
                            class_name += name + " "
                        class_name = class_name[:-1]
                        difficult = 1
                    else:
                        line_split = line.split()
                        x1, y1, x2, y2 = line_split[-4:]
                        class_name = ""
                        for name in line_split[:-4]:
                            class_name += name + " "
                        class_name = class_name[:-1]
                    if class_name not in self.class_names:
                        continue
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    cls_id = self.class_names.index(class_name) + 1
                    bbox = [x1, y1, x2 - x1, y2 - y1, difficult, str(image_id), cls_id, (x2 - x1) * (y2 - y1) - 10.0]
                    boxes_per_image.append(bbox)
            images.append(image)
            bboxes.extend(boxes_per_image)
        categories = []
        for i, cls in enumerate(self.class_names):
            category = {
                'supercategory': cls,
                'name': cls,
                'id': i + 1
            }
            categories.append(category)
        annotations = []
        for i, box in enumerate(bboxes):
            annotation = {
                'area': box[-1],
                'category_id': box[-2],
                'image_id': box[-3],
                'iscrowd': box[-4],
                'bbox': box[:4],
                'id': i
            }
            annotations.append(annotation)
        results = {
            'images': images,
            'categories': categories,
            'annotations': annotations
        }
        return results

    def preprocess_detection(self, detection_path):
        image_ids = os.listdir(detection_path)
        results = []
        for image_id in image_ids:
            lines_list = read_txt(path=os.path.join(detection_path, image_id))
            image_id = os.path.splitext(image_id)[0]
            for line in lines_list:
                if line != '':
                    line_split = line.split()
                    class_conf, x1, y1, x2, y2 = line_split[-5:]
                    class_name = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    if class_name not in self.class_names:
                        continue
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    result = {
                        "image_id": str(image_id),
                        "category_id": self.class_names.index(class_name) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(class_conf)
                    }
                    results.append(result)
        return results
