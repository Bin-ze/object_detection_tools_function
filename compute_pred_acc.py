import os
import json
import sys
import logging
import argparse
import cv2

import numpy as np

'''

计算yolo txt 标注下的gt以及pred是否匹配，
输入：gt txt  pred txt(自己生成，格式每行 'label_id,x,y,w,h')
输出：txt文件保存准确率，以及漏检图片名称和分类错误图片名称

'''
class Cumpute_pred_label:

    def __init__(self, pred_path, label_path, conf_thres=0.5, result_sava_path='./'):
        """

        :param pred_path:
        :param label_path:
        :param conf_thres:
        :param result_sava_path:
        """
        self.pred_path = pred_path
        self.label_path = label_path
        self.conf_thres = conf_thres
        self.count = 0
        self.total_gt = 0
        self.save_path = result_sava_path
        # 漏检
        self.miss = []
        self.classify_error = []

    def compute_single_img(self, pred_txt, label_txt):
        """

        :param pred_txt:
        :param label_txt:
        :return:
        """
        with open(pred_txt, 'r') as f:
            pred_result = [x.split() for x in f.read().strip().splitlines()]
        with open(label_txt, 'r') as f:
            label_result = [x.split() for x in f.read().strip().splitlines()]

        pred_result = np.array(pred_result, dtype=np.float32)

        if pred_result.shape[1] == 6:
            pred_result = pred_result[:, :-1]

        label_result = np.array(label_result, dtype=np.float32)
        # update total_gt_nums
        self.total_gt += label_result.shape[0]
        # 开始计算
        for instance in label_result:
            # 当pred的结果全部弹出，则退出循环
            if pred_result.shape[0] == 0:
                self.miss.append(pred_txt)
                break

            label_instance = instance[None, :1]
            bbox_instance = self.xywh2xyxy(instance[None, 1:])
            bbox_iou = self.bbox_iou(bbox_instance, self.xywh2xyxy(pred_result[:, 1:]))

            max_iou_index = np.argmax(bbox_iou)
            # 判断bbox是否匹配成功
            if bbox_iou[0][max_iou_index] >= self.conf_thres:
                # 判断label是否匹配成功
                if label_instance.item() == pred_result[max_iou_index].item(0):
                    self.count += 1
                    # update pre_result
                    pred_result = np.delete(pred_result, max_iou_index, axis=0)

                else:

                    pred_result = np.delete(pred_result, max_iou_index, axis=0)
                    # 添加分类错误文件名
                    self.classify_error.append(pred_txt)
            else:
                pred_result = np.delete(pred_result, max_iou_index, axis=0)
                #self.miss.append(pred_txt)
        return

    def compute(self):
        """

        :return:
        """
        label_ = sorted(os.listdir(self.pred_path))
        #label_ = sorted(os.listdir(self.label_path))
        for label in label_:
            pred_txt = os.path.join(self.pred_path, label)
            label_txt = os.path.join(self.label_path, label)
            self.compute_single_img(pred_txt, label_txt)

    def format_result(self):
        """

        :return:
        """
        total_gt = dict(total_gt=self.total_gt)
        assign_pred = dict(assign_pred=self.count)
        dict_input = dict(total_gt=total_gt, assign_pred=assign_pred, acc=round(self.count/self.total_gt, 4))
        logging.info('json formatted output')
        if os.path.isfile(self.save_path): os.remove(self.save_path)
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(dict_input, indent=4))

            f.write('\n漏检图片文件名:\n')
            for miss in set(self.miss):
                miss = miss.split('/')[-1].replace('txt','jpg')
                f.write(miss+'\n')

            f.write('错检图片文件名:\n')
            for error in set(self.classify_error):
                error = error.split('/')[-1].replace('txt', 'jpg')
                f.write(error+'\n')

        logging.info(f'statistics result save in {self.save_path}')
        print(f'total gt: {self.total_gt}')
        print(f'assign pred: {self.count}')
        print('acc: {:.3f}'.format(self.count/self.total_gt))

    @staticmethod
    def area(box):
        """
        :param box:
        :return:
        """
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    def bbox_iou(self, box1, box2):
        """
        :param box1:tensor [M,4]
        :param box2: tensor [N,4]
        :return: box1 and box2 iou
        intersection:交集
        union:并集
        """
        area1 = self.area(box1)
        area2 = self.area(box2)
        a = np.maximum(box1[:, None, :2], box2[:, :2])
        b = np.minimum(box1[:, None, 2:], box2[:, 2:])
        wh = np.where((b - a) < 0, 0, b - a)  # 如果两个box不相交，那么输出为0，不能出现分子为负数的情况
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def __call__(self, *args, **kwargs):
        self.compute()
        self.format_result()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='/object_detection/data/pinjie_data/coco/labels/test',help='gt txt save path')
    parser.add_argument('--pred_path', type=str, default='/object_detection/yolov7/runs/test/exp2/labels',help='pred result txt save path')
    parser.add_argument('--result_save_path', type=str, default='./result.txt', help='acc result and error and miss result save path')
    args = parser.parse_args()
    logging.info(args)
    compute = Cumpute_pred_label(pred_path=args.pred_path, label_path=args.label_path, result_sava_path=args.result_save_path)
    compute()