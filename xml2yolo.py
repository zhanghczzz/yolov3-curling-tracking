import os
import xml.etree.ElementTree as ET


#生成yolo格式的标注文件
def xml2txt():
    dirpath = 'data/custom/xml'  # 原来存放xml文件的目录
    newdir = 'data/custom/txt_curling'  # 修改label后形成的txt目录
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    for fp in os.listdir(dirpath):

        root = ET.parse(os.path.join(dirpath, fp)).getroot()

        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')
        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text
        for child in root.findall('object'):  # 找到图片中的所有框
            name = child.find('name').text #标签名
            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
            except ZeroDivisionError:
                print(filename, '的 width有问题')

            #red和yellow标签
            # if name == 'red':
            #     label = 0
            # else:
            #     label = 1
            label = 0

            with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
                f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
#生成train.txt，文件中包含图片文件的地址
def createTrain():
    dirpath = 'data/custom/images'
    count = 0 #计数器，用350张作为训练集
    with open('data/custom/train_curling.txt', 'a+') as f:
        for img in os.listdir(dirpath):
            if count == 350:
                break
            f.write(dirpath+'/'+img+'\n')
            count += 1

def createValid():
    dirpath = 'data/custom/images'
    list = os.listdir(dirpath)
    with open('data/custom/valid.txt', 'a+') as f:
        for img in range(300, 350):
            f.write(dirpath + '/' + list[img] + '\n')

if __name__ == '__main__':
    xml2txt()