import cv2
import time
import mmcv
import json
import os
import mmdeploy_python
import logging
import argparse
import time
import sys

import numpy as np

'''
此脚本无法直接运行，但是里面的一些函数实现值得借鉴
包括：
    如何画框，随机颜色实现，如何画label等等
    
参考：https://github.com/ultralytics/yolov5/blob/master/detect.py
借助mmdeploy实现模型部署之后的模型推理流程，实现了包括推理，画框，可视化的流程
支持单独调用，以及直接调用类自身完成
'''

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class Inference():
    def __init__(self, input_image, model_path, save_path):
        """I
        Initialize model
        Returns: model
        """
        self.names = ('banmaxian', 'bao', 'beizi', 'chuang', 'diandongche',
                  'diannao', 'dianshi', 'fushou', 'gongjiaoche', 'gou',
                  'heiban', 'honglvdeng', 'hua', 'jianpan', 'jiansudai',
                  'jingshipai', 'lajitong', 'lanqiu', 'mao', 'men', 'mianbao',
                  'ren', 'shafa', 'shidun', 'shouji', 'shuben', 'shubiao', 'taideng',
                  'taijie', 'wanju', 'weilan', 'xiaoqiche', 'xinglixiang', 'yigui',
                  'yizi', 'zhuozi', 'zixingche', 'zuqiu')
        self.model_path = model_path
        self.save_path = save_path
        self.input_image = input_image
        self.detector = mmdeploy_python.Detector(self.model_path, 'cuda', 0)
        self.path = ''

    def process_image(self, input_image):
        """
        :param input_image:
        :return:
        """
        self.img_path = input_image.split("/")[-1]
        self.im = mmcv.imread(input_image)
        result = self.detector([self.im])
        return result

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """

        :param box:
        :param label:
        :param color:
        :param txt_color:
        :return:
        """
        # 先画框
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        # 画框
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1, txt_color,
                        thickness=1, lineType=cv2.LINE_AA)

    def plot_bbox(self, result, view_img=False, imwrite=True):
        bboxes = result[0][0]
        labels = result[0][1]
        inds = bboxes[:, -1] >= 0.3
        bboxes,conf = bboxes[inds, :],bboxes[inds, -1]
        labels = labels[inds]
        for j, box in enumerate(bboxes):
            cls=labels[j]
            color = colors(cls)
            cls = self.names[cls]
            label = f'{cls} {conf[j]:.1f}'
            self.box_label(box,label,color=color)
        if view_img:
            cv2.imshow(str(self.path+self.img_path), self.im)
            cv2.waitKey(1)  # 1 millisecond
        if imwrite:
            cv2.imwrite('{}/{}'.format(self.save_path, self.img_path), self.im)

    def __call__(self):
        result = self.process_image(self.input_image)
        self.plot_bbox(result)

## define fine val test
def find_valset(val_txt_path):
    val = val_txt_path
    valset=[]
    with open(val,'r') as f:
        for img in f.readlines():
            valset.append(img.split("\n")[0])
    return valset


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='', help='infer img path')
    parser.add_argument('--model_path', type=str, default='', help='use infer model path')
    parser.add_argument('--save_path', type=str, default='', help='infer result save path')
    args = parser.parse_args()
    logging.info(args)
    # create valset list
    valset=find_valset()
    # instantiate inference class
    inf = Inference(input_image=args.img_path, model_path=args.model_path, save_path=args.save_apth)
    inf()


    # loop run
    path = ''
    img_dir = os.listdir(path)
    for img in img_dir:
        # if img in val set: infrernce   else:  continue
        #if img.split('.')[0] not in valset: continueI
        whole_img = os.path.join(path,img)
        t1 = time.time()
        inf.plot_bbox(inf.process_image(whole_img))
        #print(result)
        print('inference_time：', time.time() - t1)
