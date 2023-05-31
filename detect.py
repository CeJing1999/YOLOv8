import os
import cv2
import time
import torch
import argparse
import colorsys
import numpy as np

from torch import nn
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from nets.yolo import YOLO
from utils.util_box import non_max_suppression
from utils.util import img_to_rgb, preprocess_input, get_classes, show_info


def detect_image(
        image: Image, model: nn.Module, class_names: List, num_classes: int = 80,
        input_shape: Tuple = (640, 640), conf_thres: float = 0.25, iou_thres: float = 0.45
):
    image = img_to_rgb(image=image)  # 转换成RGB图像
    iw, ih = image.size  # 获得图像的宽高
    h, w = input_shape  # 获得目标高宽
    scale = min(w / iw, h / ih)  # 获得图像缩放比例
    nw, nh = int(iw * scale), int(ih * scale)  # 获得缩放后的图像宽高
    dx, dy = (w - nw) // 2, (h - nh) // 2  # 计算图像的偏移量

    image_data = image.resize((nw, nh), resample=Image.BICUBIC)  # 调整图像大小
    new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
    new_image.paste(image_data, box=(dx, dy))  # 将图像多余的部分加上灰条
    # 转换成ndarray ===> 归一化到0~1之间 ===> 通道提前 ===> 添加batch_size维度
    image_data = np.expand_dims(
        np.transpose(preprocess_input(image=np.array(new_image, dtype=np.float32)), axes=(2, 0, 1)), axis=0
    )

    with torch.no_grad():
        image_data = torch.from_numpy(image_data).cuda()  # 转换成torch.Tensor
        output, _ = model(image_data)  # 将图像输入网络当中进行预测
        # -------------------------------------------------- #
        #   output: x1, y1, x2, y2, conf, cls
        # -------------------------------------------------- #
        output = non_max_suppression(outputs=output, conf_thres=conf_thres, iou_thres=iou_thres)
        if len(output[0]) == 0:
            return image
        box, conf, cls = output[0][:, : 4].cpu().numpy(), output[0][:, 4].cpu().numpy(), output[0][:, 5].cpu().numpy()

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 不同类的画框设置不同的颜色
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    font = ImageFont.truetype(
        'model_data/sim_hei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype(dtype=np.int32)
    )  # 字体: 字体大小随图片大小自适应调整
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))  # 边框厚度也随图片大小自适应调整
    draw = ImageDraw.Draw(image)

    for i, class_id in enumerate(cls):
        class_name = class_names[int(class_id)]  # 类名
        class_conf = conf[i]  # 类别置信度
        x1, y1, x2, y2 = box[i]  # bounding box
        x1, y1, x2, y2 = (x1 - dx) / scale, (y1 - dy) / scale, (x2 - dx) / scale, (y2 - dy) / scale  # 调整至原图大小
        # 左上角坐标不能小于0，右下角坐标不能大于原图宽高
        x1, y1 = max(0, np.round(x1).astype(dtype=np.int32)), max(0, np.round(y1).astype(dtype=np.int32))
        x2, y2 = min(iw, np.round(x2).astype(dtype=np.int32)), min(ih, np.round(y2).astype(dtype=np.int32))

        label = '{} {:.2f}'.format(class_name, class_conf)  # 类别标签
        label_size = draw.textbbox(xy=(0, 0), text=label, font=font)  # 类别标签大小
        # 防止类别标签超出图片上边界
        text_origin = np.array([x1, y1 - label_size[3]]) if y1 >= label_size[3] else np.array([x1, y1 + 1])
        for j in range(thickness):  # 边框厚度随图片大小自适应调整
            draw.rectangle([x1 + j, y1 + j, x2 - j, y2 - j], outline=colors[int(class_id)])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size[2: 4])], fill=colors[int(class_id)])
        draw.text(text_origin, text=label, fill=(0, 0, 0), font=font)

    del draw
    return image


def detect(cfg, model: nn.Module, class_names: List, num_classes: int = 80):
    if cfg.mode == 'img':
        if os.path.isdir(cfg.source):
            image_list = os.listdir(cfg.source)  # 获取图像列表
            for image_path in tqdm(image_list):
                if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # 只检测jpg/jpeg/png格式的图片文件
                    image = Image.open(os.path.join(cfg.source, image_path))  # 读取图像
                    image = detect_image(image, model, class_names, num_classes)  # 检测图片
                    if cfg.save_flag:  # 保存检测文件
                        image.save(os.path.join(cfg.save_path, image_path), quality=95, sunsampling=0)
        else:
            assert cfg.source.lower().endswith(('.jpg', '.jpeg', '.png')), '请确保输入的图片文件是jpg/jpeg/png格式！'
            image = Image.open(cfg.source)  # 读取图像
            image = detect_image(image, model, class_names, num_classes)  # 检测图片
            if cfg.save_flag:  # 保存检测文件
                image.save(os.path.join(cfg.save_path, os.path.basename(cfg.source)), quality=95, sunsampling=0)
            image.show()  # 显示检测结果
    elif cfg.mode == 'vid':
        assert cfg.source.lower().endswith(('.mp4', '.avi')), '请确保输入的视频文件是.mp4/.avi格式！'
        video_reader = cv2.VideoCapture(cfg.source)
        video_writer = None
        if cfg.save_flag:
            video_format = 'MP4V' if cfg.source.lower().endswith('mp4') else 'XVID'
            save_fourcc = cv2.VideoWriter_fourcc(*video_format)
            save_fps = video_reader.get(cv2.CAP_PROP_FPS)
            size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(
                os.path.join(cfg.save_path, os.path.basename(cfg.source)), save_fourcc, save_fps, size
            )
        ref, frame = video_reader.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径！")
        fps = 0.0
        while True:
            t = time.time()
            ref, frame = video_reader.read()  # 读取某一帧
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转换: BGR ===> RGB（网络输入要求）
            frame = Image.fromarray(np.uint8(frame))  # 转化成Image
            frame = detect_image(frame, model, class_names, num_classes)  # 进行检测
            frame = np.array(frame)  # 转换成ndarray
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 格式转换: RGB ===> BGR（opencv要求）
            fps = (fps + (1. / (time.time() - t))) / 2
            # 图片/帧，文字内容，文字位置，字体样式，字体大小，字体颜色，字体粗细
            frame = cv2.putText(frame, "fps=%.2f" % fps, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('video', frame)
            c = cv2.waitKey(1) & 0xff
            if cfg.save_flag and video_writer is not None:
                video_writer.write(frame)
            if c == 27:
                video_reader.release()
                break
        print("视频检测完成！")
        video_reader.release()
        if cfg.save_flag:
            video_writer.release()
        cv2.destroyAllWindows()
    else:
        raise NotImplementedError('检测模式%s未实现！' % cfg.mode)


def main(cfg):
    class_names, num_classes = get_classes(cfg.classes_path)  # 获取类别
    show_info(
        mode=cfg.mode, source=cfg.source,
        save_flag=cfg.save_flag, save_path=cfg.save_path,
        model_path=cfg.model_path, classes_path=cfg.classes_path,
        input_shape=cfg.input_shape, stride=cfg.stride,
        class_names=class_names, num_classes=num_classes,
        conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres
    )

    model = YOLO(phi=cfg.phi, num_classes=num_classes, stride=cfg.stride)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict=torch.load(cfg.model_path, map_location=device))
    model = model.eval()
    print('{} model and classes loaded.'.format(cfg.model_path))
    model = nn.DataParallel(module=model)
    model = model.cuda()

    if cfg.save_flag and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    detect(cfg, model, class_names, num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='img', help='检测模式: img、vid')
    parser.add_argument('--source', type=str, default='data/zidane.jpg', help='待检测文件路径')
    parser.add_argument('--save_flag', type=bool, default=True, help='是否保存检测结果')
    parser.add_argument('--save_path', type=str, default='outs/', help='检测结果保存路径')
    parser.add_argument('--model_path', type=str, default='model_data/epoch500/best_epoch_weights.pt', help='权值文件')
    parser.add_argument('--classes_path', type=str, default='model_data/voc_classes.txt', help='类别文件')
    parser.add_argument('--input_shape', type=Tuple, default=(640, 640), help='输入图像大小')
    parser.add_argument('--stride', type=Tuple, default=(8, 16, 32), help='下采样步长')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='非极大抑制的类别置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='非极大抑制的iou置信度阈值')
    parser.add_argument('--phi', type=str, default='n', help='版本: n, s, l, m, x')
    args = parser.parse_args()
    main(cfg=args)
