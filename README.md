# `YOLOv8`

---

这是一个自定义实现的`YOLOv8`目标检测模型，官方`github`仓库位于<https://github.com/ultralytics/ultralytics.git>。

## 安装

#### 使用`conda`创建并激活虚拟环境

项目开发过程中使用的`Python`版本是`3.10`

```
conda create -n conda_env_name python=3.10  # create
conda activate conda_env_name  # activate
```

#### 安装`PyTorch`

`PyTorch`官网：<https://pytorch.org>

项目开发过程中使用的`PyTorch`版本是`1.13.0+cu116`

```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### 克隆仓库并安装依赖

```
git clone https://github.com/CeJing1999/YOLOv8.git  # clone
cd YOLOv8
pip install -r requirement.txt  # install
```

## 训练

```
python train.py --batch_size 8 --classes_path model_data/voc_classes.txt --train_path dataset/VOC/train.txt --val_path dataset/VOC/val.txt
```

## 推理

```
python detect.py --mode img --source data/street.jpg --model_path model_data/xxx.pt --classes_path model_data/voc_classes.txt
                        vid          data/xxx.mp4
```

## 数据集

以`coco`和`VOC`数据集为例，数据集的组织结构如下所示：

```
YOLOv8
└── dataset
    └── coco
    |   ├── images
    |   │   ├── train2017
    |   │   │   ├── xxx.jpg
    |   │   │   └── ......
    |   │   ├── val2017
    |   │   │   ├── xxx.jpg
    |   │   │   └── ......
    |   |── labels
    |   |   ├── train2017
    |   |   │   ├── xxx.txt
    |   |   │   └── ......
    |   |   └── val2017
    |   |       ├── xxx.txt
    |   |       └── ......
    |   |—— train.txt  # 存放训练集图片路径，程序将会读取该文件以获得训练数据
    |   └── val.txt  # 存放验证集图片路径，程序将会读取该文件以获得验证数据
    └── VOC
        ├── images
        │   ├── test2007
        │   │   ├── xxx.jpg
        │   │   └── ......
        │   ├── train2007
        │   │   ├── xxx.jpg
        │   │   └── ......
        │   ├── train2012
        │   │   ├── xxx.jpg
        │   │   └── ......
        │   ├── val2007
        │   │   ├── xxx.jpg
        │   │   └── ......
        │   └── val2012
        │       ├── xxx.jpg
        │       └── ......
        |── labels
        |   ├── test2007
        |   │   ├── xxx.txt
        |   │   └── ......
        |   ├── train2007
        |   │   ├── xxx.txt
        |   │   └── ......
        |   ├── train2012
        |   │   ├── xxx.txt
        |   │   └── ......
        |   ├── val2007
        |   │   ├── xxx.txt
        |   │   └── ......
        |   └── val2012
        |       ├── xxx.txt
        |       └── ......
        |—— train.txt  # 存放训练集图片路径，程序将会读取该文件以获得训练数据
        └── val.txt  # 存放验证集图片路径，程序将会读取该文件以获得验证数据
```

