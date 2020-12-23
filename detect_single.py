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

def getitem(original_img, img_size):
    img = transforms.ToTensor()(cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0) #utils.datasets中的一个方法
    # Resize
    img = resize(img, img_size)
    return img.unsqueeze(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom-curling.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/curling.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes_curling.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--image_path", type=str, default="data/samples2/test1.PNG", help="path to image")
    opt = parser.parse_args()
    print(opt)

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

    image_path = opt.image_path
    img = cv2.imread(image_path) #opencv格式的原始图像
    print(img.shape)
    input = Variable(getitem(img, opt.img_size).cuda()) #将原始图像进行处理
    classes = load_classes(opt.class_path)  # 加载类别

    print("\nPerforming object detection:")
    prev_time = time.time()
    # Get detections
    with torch.no_grad():
        detections = model(input)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0] #返回的是个列表，所以取第一个

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\tInference Time: %s" % (inference_time))
    # Iterate through images and save plot of detections 格式(B, G, R)
    colors = [(0, 0, 255), (0, 215, 255)]
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        print(detections.shape)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_conf.item() < 0.9:
                continue
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            print(conf)

            color = colors[int(cls_pred)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            # Add the bbox to the plot
            # Add label
            cv2.rectangle(img, (x1, y1), (x2 + 40, y1 - 20), color, thickness=-1)
            cv2.putText(
                img,
                classes[int(cls_pred)],
                (x1, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                thickness=2
            )
        # Save generated image with detections
    filename = image_path.split("/")[-1].split(".")[0]
    cv2.imwrite(f"output/{filename}.png", img)
