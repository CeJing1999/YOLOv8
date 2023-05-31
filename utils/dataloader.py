import torch
import numpy as np

from typing import List
from torch.utils.data import Dataset, DataLoader


class YoloDataLoader(DataLoader):
    """
        YOLO DataLoader
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuttle: 是否打乱
            num_workers: 加载数据的线程数
            pin_memory: 是否在返回张量之前将张量复制到设备（CUDA）内存中
            drop_last: 是否删除最后一个不完整的批次
    """

    def __init__(
            self, dataset: Dataset, batch_size: int = 1, shuttle: bool = True,
            num_workers: int = 0, pin_memory: bool = True, drop_last: bool = True
    ):
        super(YoloDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuttle, num_workers=num_workers,
            collate_fn=self.dataset_collate, pin_memory=pin_memory, drop_last=drop_last
        )

    @staticmethod
    def dataset_collate(batch: List):
        images, boxes = [], []
        max_dim = max([len(box) for _, box in batch])
        for image, box in batch:
            temp = np.zeros((max_dim, 5), dtype=np.float32)
            if len(box) > 0:
                temp[: len(box)] = box
            images.append(image)
            boxes.append(temp)
        images = torch.from_numpy(np.array(images)).type(dtype=torch.FloatTensor)
        boxes = torch.from_numpy(np.array(boxes)).type(dtype=torch.FloatTensor)
        return images, boxes
