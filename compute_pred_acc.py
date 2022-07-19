import os
import json
import sys
import logging
import argparse

import numpy as np

'''

计算yolo txt 标注下的gt以及pred是否匹配，
输入：gt txt  pred txt (自己生成，格式每行 'label_id,x,y,w,h,conf' )
注：因为保存的txt默认为已经根据置信度conf项可有可无
输出：txt文件保存准确率，以及漏检图片名称和分类错误图片名称

测试情况：
1 只有错检 true
2 只有漏检 true
3 既有错检又有漏检 true

logic：
读取每张测试图片的gt_txt,pred_txt
对于这样一对文件(对应单张图片的pred以及gt)，进行匹配计算：
1 对于图片上的每一个gt，使用IOU匹配对应的pred,如果匹配成功，进入下一步(查看类别是否匹配),匹配失败，则记录漏检
2 对于第一步匹配成功的pred,查看类别是否匹配，不匹配，则记录错检
3 pop出已经匹配的pred标签，防止影响结果，接着进入下一次循环
        
'''
class Cumpute_pred_acc:

    def __init__(self, pred_path, label_path, iou_thres=0.5, result_sava_path='./'):
        """

        :param pred_path:
        :param label_path:
        :param conf_thres:
        :param result_sava_path:
        """
        self.pred_path = pred_path
        self.label_path = label_path
        self.iou_thres = iou_thres
        self.count = 0
        self.total_gt = 0
        self.save_path = result_sava_path
        # 漏检
        self.miss = []
        # 错检
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
            if bbox_iou[0][max_iou_index] >= self.iou_thres:
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
                # pred_result = np.delete(pred_result, max_iou_index, axis=0)
                self.miss.append(pred_txt)
        return

    def compute(self):
        """

        :return:
        """
        label_ = sorted(os.listdir(self.pred_path))

        # 测试图片总数
        self.total_img_nums = len(label_)

        #label_ = sorted(os.listdir(self.label_path))
        for label in label_:
            pred_txt = os.path.join(self.pred_path, label)
            label_txt = os.path.join(self.label_path, label)
            self.compute_single_img(pred_txt, label_txt)

    def format_result(self):
        """

        :return:
        """

        # instance level统计结果
        instance_level = dict(total_gt_nums=self.total_gt, match_pred_nums=self.count,
                              instance_level_acc=round(self.count/self.total_gt, 4))
        # 计算错误图片
        error_img_nums = len(set(self.miss + self.classify_error))
        # image level 统计结果
        img_level = dict(total_img_nums=self.total_img_nums, correct_img_nums=self.total_img_nums - error_img_nums,
                         img_level_acc=round((self.total_img_nums - error_img_nums)/self.total_img_nums, 4))

        dict_input = dict(instance_level=instance_level,
                          img_level=img_level)

        logging.info('json formatted output')
        if os.path.isfile(self.save_path): os.remove(self.save_path)

        # 结果写入
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(dict_input, indent=4))

            f.write('\n___漏检图片文件名:___\n')
            for miss in sorted(set(self.miss)):
                miss = miss.split('/')[-1].replace('txt','jpg')
                f.write(miss+'\n')

            f.write('___错检图片文件名:___\n')
            for error in sorted(set(self.classify_error)):
                error = error.split('/')[-1].replace('txt', 'jpg')
                f.write(error+'\n')


        logging.info(f'statistics result save in {self.save_path}')
        logging.info(f'total gt: {self.total_gt}')
        logging.info(f'match pred: {self.count}')
        logging.info(f'total img: {self.total_img_nums}')
        logging.info(f'correct_img: {self.total_img_nums - error_img_nums}')
        logging.info('instance_level_acc: {:.4f}'.format(self.count/self.total_gt))
        logging.info('img_level_acc: {:.4f}'.format((self.total_img_nums - error_img_nums)/self.total_img_nums))

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

    def __call__(self):
        self.compute()
        self.format_result()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='/home/guozebin/object_detection/data/pinjie_data/coco/labels/test',help='gt txt save path')
    parser.add_argument('--pred_path', type=str, default='/home/guozebin/object_detection/yolov7/runs/detect/exp2/labels',help='pred result txt save path')
    parser.add_argument('--result_save_path', type=str, default='./result.txt', help='acc result and error and miss result save path')
    args = parser.parse_args()
    logging.info(args)
    compute = Cumpute_pred_acc(pred_path=args.pred_path, label_path=args.label_path, result_sava_path=args.result_save_path)
    compute()