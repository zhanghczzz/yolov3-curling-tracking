# YOLOv3-Curling-Tracking

基于 Yolov3 + Deepsort 实现的冰壶识别与跟踪，Yolov3框架采用 https://github.com/eriklindernoren/PyTorch-YOLOv3 ，数据集视频来自YouTube，手动分割标注。原文： https://zhanghc.site/archives/yolo-curling-track 。

## Installation

##### Clone and install requirements

    $ git clone git@github.com:zhhaochen/yolov3-curling-tracking.git
    $ cd PyTorch-YOLOv3/
    $ pip install -r requirements.txt

##### Download pretrained weights

    # yolov3
    cd weights
    sh download_weights.sh
    # curling
    https://drive.google.com/file/d/1FC5xLvL-jQSNNBXKI-ocZL19bc9xxiu5/view?usp=sharing

##### Download dataset

    https://drive.google.com/file/d/1qkbhFc-BeE348NCQKQ_NALYzaUZeCYaj/view?usp=sharing

##### Project structure

```shell
${Curling_ROOT}
├── checkpoints         		#预训练冰壶检测权重
├── config             			#yolov3模型配置文件
├── data               			#冰壶数据集和标注
├── deep_sort       			#基于deepsort的跟踪算法模块
├── test               			#对detect_model中的模块的测试文件
├── utils              			#yolov3一些相关函数
├── weights            			#基本yolov3预训练权重
├── darknet.py         			#darknet模型
├── detect.py					#原基于coco数据集的检测文件，作为示例
├── detect_folder.py   			#文件夹图片检测示例文件，在detect.py基础上使用cv2画图
├── detect_single.py   			#基于单张图片的冰壶检测
├── detect_webcam.py   			#基于视频的冰壶检测
├── detect_webcam_deepsort.py   #基于视频的冰壶检测，并使用deepsort跟踪
├── detect_webcam_track.py      #基于视频的冰壶检测，并使用opencv自带跟踪
├── models.py           		#yolov3模型
├── spilt_video.py         		#视频抽帧切割，用于数据集制作
├── train_curling.py      		#训练冰壶检测
├── xml2yolo.py					#将labelImg标注转为yolo标注
├── requirements.txt
```

## Run

修改各测试文件配置，直接运行即可，如图为基于deepsort的跟踪检测。

![](https://zhanghc-blog-pic.oss-cn-beijing.aliyuncs.com/blog_pic/yolo_deepsort_curling.gif)