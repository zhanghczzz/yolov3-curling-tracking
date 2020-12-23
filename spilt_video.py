import cv2
import os

def video_split(video_path,save_path):
    cap=cv2.VideoCapture(video_path)
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
    frame_count = 482
    fps = 0
    #每25帧抽一张
    while (True):
        ret, frame = cap.read()
        fps = fps+1
        if ret is False:
            break
        if fps % 10 != 0:
            continue
        else:
            fps = 0
            cv2.imwrite(save_path + "/" + str('%06d' % frame_count) + '.png', frame)
            frame_count = frame_count + 1
    # frames = cap.get(7) #总帧数
    cap.release()
    print(video_path + ' frame_count: ' + str(frame_count))
    # print(len(all_frames))

def rename_img():

    path = 'f:/curling/outpu/'
    # 获取该目录下所有文件，存入列表中
    f = os.listdir(path)

    frame_count = 0
    for i in f:
        oldname = path + i
        newname = path + str('%04d' % frame_count) + '.png'
        frame_count += 1

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)

def save_video(path):
    # filelist = [str('%06d' % i) + '.png' for i in range(353)]
    filelist = os.listdir(path)
    print(filelist)
    writer = None
    for item in filelist:
        # print(item)
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter("output/binghu.avi", fourcc, 25,
                                        (img.shape[1], img.shape[0]), True)
            if writer is not None:
                writer.write(img)


if __name__ == "__main__":
    # video_path = 'F:/curling/input/curling10.mp4'
    # save_path = 'F:/curling/outpu/'
    # video_split(video_path, save_path)
    # rename_img()
    path = 'output/curling'
    save_video(path)