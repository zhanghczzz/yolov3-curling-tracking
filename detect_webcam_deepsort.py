from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.autograd import Variable
from deep_sort import DeepSort

import cv2

def getitem2(original_img, img_size):
    img = transforms.ToTensor()(cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0) #utils.datasets中的一个方法
    # Resize
    img = resize(img, img_size)
    return img.unsqueeze(0)

def bbox_rel(x1, y1, x2, y2):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x_c, y_c, w, h
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom-curling.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/curling.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes_curling.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--video_path", type=str, default="data/video/curling5.mp4", help="path to video")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # 加载类别
    # Iterate through images and save plot of detections 格式(B, G, R),不要写反
    colors = [(0, 0, 255), (0, 215, 255)]

    prev_time = time.time()
    cap = cv2.VideoCapture(opt.video_path)
    print('start')
    frame_cout = 0
    out = None
    while True:
        ret, img = cap.read() #opencv格式的原始图像
        # print(img.shape)
        if not ret:
            break
        input = Variable(getitem2(img, opt.img_size).cuda()) #将原始图像进行处理
        # Get detections
        with torch.no_grad():
            detections = model(input)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0] #返回的是个列表，所以取第一个
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            bbox_xywh = []
            confs = []

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if cls_conf.item() < 0.9:
                    continue
                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                x_c, y_c, bbox_w, bbox_h = bbox_rel(x1, y1, x2, y2)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([cls_conf.item()])
                outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)), img)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    img = draw_boxes(img, bbox_xyxy, identities)
        frame_cout +=1
        if out is None:
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter("output/deepsort_1.mp4", fourcc, 25,(img.shape[1], img.shape[0]), True) #注意这个shape一定要是真实的，不然生成不了视频
        if out is not None:
            out.write(img)
    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    print("\tInference FPS: %s" % (frame_cout/inference_time.total_seconds()))
    cap.release()
    out.release()
