import cv2
import random
import numpy as np

from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset

from utils.util_box import xywh2xyxy
from utils.util import read_txt, img_to_rgb, preprocess_input


class YoloDataset(Dataset):
    """
        YOLO Dataset
        Args:
            data: 数据集路径列表
            input_shape: 输入图像大小
            num_classes: 类别数
            epoch_length: 迭代次数
            mosaic: 是否执行mosaic数据增强
            mosaic_prob: 执行mosaic数据增强的概率
            mixup: 是否执行mixup数据增强
            mixup_prob: 执行mixup数据增强的概率
            special_aug_ratio: 执行特殊数据增强的比率
            train: 是否为训练数据集
    """

    def __init__(
            self, data: List, input_shape: Tuple = (640, 640), num_classes: int = 80, epoch_length: int = 100,
            mosaic: bool = True, mosaic_prob: float = 0.5, mixup: bool = True, mixup_prob: float = 0.5,
            special_aug_ratio: float = 0.7, train: bool = True
    ):
        super(YoloDataset, self).__init__()
        self.data = data
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.special_aug_ratio = special_aug_ratio
        self.train = train
        self.epoch_now = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.mosaic and self.rand() < self.mosaic_prob \
                and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = random.sample(self.data, k=3)
            lines.append(self.data[item])
            random.shuffle(lines)
            image, box = self.get_data_with_mosaic(image_paths=lines, input_shape=self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = random.sample(self.data, k=1)
                image2, box2 = self.get_data(image_path=lines[0], input_shape=self.input_shape, train=self.train)
                image, box = self.get_data_with_mixup(image1=image, box1=box, image2=image2, box2=box2)
        else:
            image, box = self.get_data(image_path=self.data[item], input_shape=self.input_shape, train=self.train)
        image = np.transpose(preprocess_input(image=np.array(image, dtype=np.float32)), axes=(2, 0, 1))
        box = np.array(box, dtype=np.float32)
        return image, box

    @staticmethod
    def rand(a: float = 0, b: float = 1):
        return np.random.rand() * (b - a) + a

    def get_data(
            self, image_path: str, input_shape: Tuple = (640, 640), train: bool = True,
            jitter: float = 0.3, hue: float = 0.1, sat: float = 0.7, val: float = 0.4
    ):
        image = Image.open(image_path)  # 读取图像
        img_to_rgb(image=image)  # 转换成RGB图像
        iw, ih = image.size  # 获得图像的宽高
        h, w = input_shape  # 获得目标高宽
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        data = read_txt(path=label_path)
        box = np.array([np.array(list(map(float, box.split(' ')))) for box in data])  # 获得预测框
        box[:, [1, 3]], box[:, [2, 4]] = box[:, [1, 3]] * iw, box[:, [2, 4]] * ih
        box[:, 1: 5] = xywh2xyxy(x=box[:, 1: 5])
        if not train:
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            dx, dy = (w - nw) // 2, (h - nh) // 2
            image = image.resize((nw, nh), resample=Image.BICUBIC)
            new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
            new_image.paste(image, box=(dx, dy))  # 将图像多余的部分加上灰条
            image_data = np.array(new_image, dtype=np.float32)
            if len(box) > 0:  # 对真实框进行调整
                np.random.shuffle(box)
                box[:, [1, 3]], box[:, [2, 4]] = box[:, [1, 3]] * nw / iw + dx, box[:, [2, 4]] * nh / ih + dy
                box[:, 1: 3][box[:, 1: 3] < 0], box[:, 3][box[:, 3] > w], box[:, 4][box[:, 4] > h] = 0, w, h
                box_w, box_h = box[:, 3] - box[:, 1], box[:, 4] - box[:, 2]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 丢弃无效框
            return image_data, box
        new_ar = iw / ih * self.rand(a=1 - jitter, b=1 + jitter) / self.rand(a=1 - jitter, b=1 + jitter)
        scale = self.rand(a=0.25, b=2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), resample=Image.BICUBIC)  # 对图像进行缩放并且进行长和宽的扭曲
        dx, dy = int(self.rand(a=0, b=w - nw)), int(self.rand(a=0, b=h - nh))
        new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
        new_image.paste(image, box=(dx, dy))  # 将图像多余的部分加上灰条
        image = new_image
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 翻转图像
        image_data = np.array(image, dtype=np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1  # 对图像进行色域变换，计算色域变换的参数
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))  # 将图像转到HSV上
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype=dtype)
        lut_sat = np.clip(x * r[1], a_min=0, a_max=255).astype(dtype=dtype)
        lut_val = np.clip(x * r[2], a_min=0, a_max=255).astype(dtype=dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))  # 应用变换
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        if len(box) > 0:  # 对真实框进行调整
            np.random.shuffle(box)
            box[:, [1, 3]], box[:, [2, 4]] = box[:, [1, 3]] * nw / iw + dx, box[:, [2, 4]] * nh / ih + dy
            if flip:
                box[:, [1, 3]] = w - box[:, [3, 1]]
            box[:, 1: 3][box[:, 1: 3] < 0], box[:, 3][box[:, 3] > w], box[:, 4][box[:, 4] > h] = 0, w, h
            box_w, box_h = box[:, 3] - box[:, 1], box[:, 4] - box[:, 2]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    def get_data_with_mosaic(
            self, image_paths: List, input_shape: Tuple = (640, 640),
            jitter: float = 0.3, hue: float = 0.1, sat: float = 0.7, val: float = 0.4
    ):
        h, w = input_shape
        min_offset_x, min_offset_y = self.rand(a=0.3, b=0.7), self.rand(a=0.3, b=0.7)
        image_datas, box_datas = [], []
        index = 0
        for image_path in image_paths:
            image = Image.open(image_path)  # 打开图片
            image = img_to_rgb(image=image)
            iw, ih = image.size  # 图片的大小
            label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
            data = read_txt(path=label_path)
            box = np.array([np.array(list(map(float, box.split(' ')))) for box in data])  # 保存框的位置
            box[:, [1, 3]], box[:, [2, 4]] = box[:, [1, 3]] * iw, box[:, [2, 4]] * ih
            box[:, 1: 5] = xywh2xyxy(x=box[:, 1: 5])
            flip = self.rand() < 0.5
            if flip and len(box) > 0:  # 是否翻转图片
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [1, 3]] = iw - box[:, [3, 1]]
            new_ar = iw / ih * self.rand(a=1 - jitter, b=1 + jitter) / self.rand(a=1 - jitter, b=1 + jitter)
            scale = self.rand(a=0.4, b=1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), resample=Image.BICUBIC)  # 对图像进行缩放并且进行长和宽的扭曲
            dx, dy = 0, 0
            if index == 0:  # 将图片进行放置，分别对应四张分割图片的位置
                dx, dy = int(w * min_offset_x) - nw, int(h * min_offset_y) - nh
            elif index == 1:
                dx, dy = int(w * min_offset_x) - nw, int(h * min_offset_y)
            elif index == 2:
                dx, dy = int(w * min_offset_x), int(h * min_offset_y)
            elif index == 3:
                dx, dy = int(w * min_offset_x), int(h * min_offset_y) - nh
            new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
            new_image.paste(image, box=(dx, dy))
            image_data = np.array(new_image)
            index = index + 1
            box_data = []
            if len(box) > 0:  # 对box进行重新处理
                np.random.shuffle(box)
                box[:, [1, 3]], box[:, [2, 4]] = box[:, [1, 3]] * nw / iw + dx, box[:, [2, 4]] * nh / ih + dy
                box[:, 1: 3][box[:, 1: 3] < 0], box[:, 3][box[:, 3] > w], box[:, 4][box[:, 4] > h] = 0, w, h
                box_w, box_h = box[:, 3] - box[:, 1], box[:, 4] - box[:, 2]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[: len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)
        cut_x = int(w * min_offset_x)
        cut_y = int(h * min_offset_y)
        new_images = np.zeros((h, w, 3))
        new_images[: cut_y, : cut_x, :] = image_datas[0][: cut_y, : cut_x, :]  # 将图片分割，放在一起
        new_images[cut_y:, : cut_x, :] = image_datas[1][cut_y:, : cut_x, :]
        new_images[cut_y:, cut_x:, :] = image_datas[2][cut_y:, cut_x:, :]
        new_images[: cut_y, cut_x:, :] = image_datas[3][: cut_y, cut_x:, :]
        new_images = np.array(new_images, dtype=np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1  # 对图像进行色域变换，计算色域变换的参数
        hue, sat, val = cv2.split(cv2.cvtColor(new_images, cv2.COLOR_RGB2HSV))  # 将图像转到HSV上
        dtype = new_images.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype=dtype)
        lut_sat = np.clip(x * r[1], a_min=0, a_max=255).astype(dtype=dtype)
        lut_val = np.clip(x * r[2], a_min=0, a_max=255).astype(dtype=dtype)
        new_images = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))  # 应用变换
        new_images = cv2.cvtColor(new_images, cv2.COLOR_HSV2RGB)
        new_boxes = self.merge_boxes(boxes=box_datas, cut_x=cut_x, cut_y=cut_y)  # 对框进行进一步的处理
        return new_images, new_boxes

    @staticmethod
    def merge_boxes(boxes: List[np.ndarray], cut_x: int, cut_y: int):
        merge_box = []
        for i in range(len(boxes)):
            for box in boxes[i]:
                temp_box = []
                x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
                if i == 0:
                    if x1 > cut_x or y1 > cut_y:
                        continue
                    if x1 <= cut_x <= x2:
                        x2 = cut_x
                    if y1 <= cut_y <= y2:
                        y2 = cut_y
                if i == 1:
                    if x1 > cut_x or y2 < cut_y:
                        continue
                    if x1 <= cut_x <= x2:
                        x2 = cut_x
                    if y1 <= cut_y <= y2:
                        y1 = cut_y
                if i == 2:
                    if x2 < cut_x or y2 < cut_y:
                        continue
                    if x1 <= cut_x <= x2:
                        x1 = cut_x
                    if y1 <= cut_y <= y2:
                        y1 = cut_y
                if i == 3:
                    if x2 < cut_x or y1 > cut_y:
                        continue
                    if x1 <= cut_x <= x2:
                        x1 = cut_x
                    if y1 <= cut_y <= y2:
                        y2 = cut_y
                temp_box.append(box[0])
                temp_box.append(x1)
                temp_box.append(y1)
                temp_box.append(x2)
                temp_box.append(y2)
                merge_box.append(temp_box)
        return merge_box

    @staticmethod
    def get_data_with_mixup(image1: np.ndarray, box1: np.ndarray, image2: np.ndarray, box2: np.ndarray):
        new_image = np.array(image1, dtype=np.float32) * 0.5 + np.array(image2, dtype=np.float32) * 0.5
        if len(box1) == 0:
            new_box = box2
        elif len(box2) == 0:
            new_box = box1
        else:
            new_box = np.concatenate([box1, box2], axis=0)
        return new_image, new_box
