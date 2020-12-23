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

import cv2

def getitem2(original_img, img_size):
    img = transforms.ToTensor()(cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0) #utils.datasets中的一个方法
    # Resize
    img = resize(img, img_size)
    return img.unsqueeze(0)

#不同tracker的创建示例，并没有用到，直接用了CRST
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker

'''
stream: cv读到的视频流
model: 检测模型
'''
def intermediate_detections(stream, model):
    _, img = stream.read()
    input = Variable(getitem2(img, opt.img_size).cuda())  # 将原始图像进行处理
    with torch.no_grad():
        detections = model(input)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]  # 返回的是个列表，所以取第一个

    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        trackers_dict = dict()
        colors = []
        for idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            if cls_conf.item() < 0.9:
                continue
            '''
            BOOSTING Tracker：速度较慢，并且表现不好
            MIL Tracker：比上一个追踪器更精确，但是失败率比较高。（最低支持OpenCV 3.0.0）
            KCF Tracker：比BOOSTING和MIL都快，但是在有遮挡的情况下表现不佳。（最低支持OpenCV 3.1.0）
            CSRT Tracker：比KCF稍精确，但速度不如后者。（最低支持OpenCV 3.4.2）
            MedianFlow Tracker：在报错方面表现得很好，但是对于快速跳动或快速移动的物体，模型会失效。（最低支持OpenCV 3.0.0）
            TLD Tracker：我不确定是不是OpenCV和TLD有什么不兼容的问题，但是TLD的误报非常多，所以不推荐。（最低支持OpenCV 3.0.0
            MOSSE Tracker：速度真心快，但是不如CSRT和KCF的准确率那么高，如果追求速度选它准没错。（最低支持OpenCV 3.4.1）
            GOTURN Tracker：这是OpenCV中唯一一深度学习为基础的目标检测器。它需要额外的模型才能运行，本文不详细讲解。（最低支持OpenCV 3.2.0）
            '''
            colors.append((random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)))
            trackers_dict[idx] = cv2.TrackerMedianFlow_create()
            trackers_dict[idx].init(img, (x1, y1, x2-x1, y2-y1))
        return stream, detections, trackers_dict, colors
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom-curling.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/curling.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes_curling.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--video_path", type=str, default="data/video/track_3.mp4", help="path to video")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # 加载类别
    # Iterate through images and save plot of detections 格式(B, G, R),不要写反
    # colors = [(0, 0, 255), (0, 215, 255)]

    prev_time = time.time()
    cap = cv2.VideoCapture(opt.video_path)
    print('start')
    frame_cout = 0
    out = None
    cap, detections, trackers_dict, colors = intermediate_detections(cap, model)
    while True:
        ret, img = cap.read() #opencv格式的原始图像
        # print(img.shape)
        if not ret:
            break

        timer = cv2.getTickCount()
        if detections is not None:
            for obj, tracker in trackers_dict.items():
                ok, bbox = tracker.update(img)
                if ok:
                    detections[obj][0] = bbox[0]
                    detections[obj][1] = bbox[1]
                    detections[obj][2] = bbox[0] + bbox[2]
                    detections[obj][3] = bbox[1] + bbox[3]
                    color = colors[obj]
                    cv2.rectangle(img, (detections[obj][0], detections[obj][1]), (detections[obj][2], detections[obj][3]), color, 4)
                    # cv2.rectangle(img, (detections[obj][0], detections[obj][1]), (detections[obj][2] + 40, detections[obj][1] - 20), color, thickness=-1)
                    # cv2.putText(
                    #     img,
                    #     classes[int(detections[obj][6])],
                    #     (detections[obj][0], detections[obj][1]),
                    #     cv2.FONT_HERSHEY_PLAIN,
                    #     2,
                    #     (255, 255, 255),
                    #     thickness=2
                    # )
                else:
                    print('Failed to track ', obj)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        frame_cout +=1
        if out is None:
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter("output/track_2.mp4", fourcc, 25,(img.shape[1], img.shape[0]), True) #注意这个shape一定要是真实的，不然生成不了视频
        if out is not None:
            out.write(img)
    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    print("\tInference FPS: %s" % (frame_cout/inference_time.total_seconds()))
    cap.release()
    out.release()
