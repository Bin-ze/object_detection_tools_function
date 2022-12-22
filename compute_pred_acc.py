import os
import json
import sys
import logging
import argparse
import cv2

import numpy as np
import prettytable as pt
'''
一种自定义的评价指标：
instance_level_acc：评估目标检测的准确率
img_level_acc: 评估所有样本都检测正确的图像占图像总个数的比例

计算yolo txt 标注下的gt以及pred是否匹配，
输入：gt txt  pred txt (自己生成，格式每行 'label_id,x,y,w,h,conf' )
注：因为保存的txt默认为已经根据置信度conf项可有可无
输出：txt文件保存准确率，以及漏检图片名称和分类错误图片名称

测试情况：
1 只有错检 true
2 只有漏检 true
3 既有错检又有漏检 true
4 不匹配的多检 true

logic：
读取每张测试图片的gt_txt,pred_txt
对于这样一对文件(对应单张图片的pred以及gt)，进行匹配计算：
1 对于图片上的每一个gt，使用IOU匹配对应的pred,如果匹配成功，进入下一步(查看类别是否匹配),匹配失败，则记录漏检
2 对于第一步匹配成功的pred,查看类别是否匹配，不匹配，则记录错检
3 pop出已经匹配的pred标签，防止影响结果，接着进入下一次循环

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

class Cumpute_pred_acc:

    def __init__(self, pred_path, label_path, iou_thres=0.4, result_sava_path='./', classes_dict=None, mask=None):
        """

        :param pred_path:
        :param label_path:
        :param iou_thres:
        :param result_sava_path:
        :param classes_dict: ZH_dict output
        :param mask: In the display result is to remove the corresponding category
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
        #
        if classes_dict is not None:
            self.classes_dict = {k:v for v, k in classes_dict.items()}
        else:
            self.classes_dict = None

        # mask
        self.mask = mask
    
    # 添加了类别过滤功能，可以计算需要的类别的准确率
    def mask_category(self, pred, label):
        new_label = []
        new_pred = []
        for i in label:
            if i[0] not in self.mask:
                new_label.append(i)
        for j in pred:
            if j[0] not in self.mask:
                new_pred.append(j)

        if not (new_pred and new_label):
            if new_pred == [] and new_label != []:
                new_pred = np.array([])
                return new_pred, np.stack(new_label)

            if new_label == []:
                new_label = np.array([])

                return np.stack(new_pred), new_label

        return np.stack(new_pred), np.stack(new_label)


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

        # Remove categories that do not need to be calculated
        if isinstance(self.mask, list):
            pred_result, label_result = self.mask_category(pred_result, label_result)

        # update total_gt_nums
        self.total_gt += label_result.shape[0]
        # 开始计算
        for instance in label_result:
            # 当pred的结果全部弹出，则退出循环
            if pred_result.shape[0] == 0:
                self.add_description_miss(pred_txt, instance[None, :1])
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
                    self.add_description_error(pred_txt, label_instance, pred_result[max_iou_index].item(0))
                    pred_result = np.delete(pred_result, max_iou_index, axis=0)
                    # 添加分类错误文件名
                    self.classify_error.append(pred_txt)
            else:
                # pred_result = np.delete(pred_result, max_iou_index, axis=0)
                self.add_description_miss(pred_txt, label_instance)

                self.miss.append(pred_txt)
        # 如果遍历完gt,此时还有pred没有弹出，说明存在不匹配的错误检测
        if pred_result.shape[0] != 0:
            #self.classify_error.append(pred_txt)
            pass
        return

    def add_description_miss(self, pred_txt, label_instance):
        """

        """
        if self.classes_dict is not None:
            info_img = pred_txt.split('/')[-1].replace('txt', 'jpg')
            info = f'{info_img}: {self.classes_dict[int(label_instance.item())]} 被漏检'
            logging.info(info)
        else:
            info_img = pred_txt.split('/')[-1].replace('txt', 'jpg')
            info = f'{info_img}: {str(int(label_instance.item()))} 被漏检'
            logging.info(info)
        return

    def add_description_error(self, pred_txt, label_instance, pred_label):
        """

        """
        if self.classes_dict is not None:
            info_img = pred_txt.split('/')[-1].replace('txt', 'jpg')
            info = f'{info_img}: {self.classes_dict[int(label_instance.item())]} 被错检为 {self.classes_dict[int(pred_label)]}'
            logging.info(info)
        else:
            info_img = pred_txt.split('/')[-1].replace('txt', 'jpg')
            info = f'{info_img}: {str(int(label_instance.item()))} 被错检为 {int(pred_label)}'
            logging.info(info)
        return

    def compute(self):
        """

        :return:
        """
        label_ = sorted(os.listdir(self.pred_path))

        # 测试图片总数
        self.total_img_nums = len(label_)

        # label_ = sorted(os.listdir(self.label_path))
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
                              instance_level_acc=round(self.count / self.total_gt, 4))
        # 计算错误图片
        error_img_nums = len(set(self.miss + self.classify_error))
        # image level 统计结果
        img_level = dict(total_img_nums=self.total_img_nums, correct_img_nums=self.total_img_nums - error_img_nums,
                         img_level_acc=round((self.total_img_nums - error_img_nums) / self.total_img_nums, 4))

        dict_input = dict(instance_level=instance_level,
                          img_level=img_level,
                          Missing_detection_rate=len(set(self.miss)) / self.total_img_nums,
                          False_detection_rate=(len(set(self.classify_error))) / self.total_img_nums)

        logging.info('json formatted output')
        if os.path.isfile(self.save_path): os.remove(self.save_path)

        # 结果写入
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(dict_input, indent=4))

            f.write('\n___漏检图片文件名:___\n')
            for miss in sorted(set(self.miss)):
                miss = miss.split('/')[-1].replace('txt', 'jpg')
                f.write(miss + '\n')

            f.write('___错检图片文件名:___\n')
            for error in sorted(set(self.classify_error)):
                error = error.split('/')[-1].replace('txt', 'jpg')
                f.write(error + '\n')

        tb = pt.PrettyTable()
        tb.field_names = ['statistics result save path', 'total gt', 'match pred', 'total img', 'correct_img', 'instance_level_acc', 'img_level_acc', 'Missing detection rate', 'False detection rate']
        tb.add_row([self.save_path, self.total_gt, self.count, self.total_img_nums, self.total_img_nums - error_img_nums, round(self.count / self.total_gt, 4),
                    round((self.total_img_nums - error_img_nums) / self.total_img_nums, 4), round((len(set(self.miss))) / self.total_img_nums, 4),round(len(set(self.classify_error)) / self.total_img_nums, 4)])
        print(tb)
        return

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

    @staticmethod
    # The image size is fixed here. If the size of each image is different, you need to modify the value here
    def denorm_yolo(box, wh=(1000, 720)):

        bbox = box.copy()
        bbox[:, 1] *= wh[0]
        bbox[:, 2] *= wh[1]
        bbox[:, 3] *= wh[0]
        bbox[:, 4] *= wh[1]

        return bbox

    def plot_bbox(self, txt_path, img_path, save_path='../visualization/', view_img=False, imwrite=True):
        with open(txt_path, 'r') as f:
            plot_label = [x.split() for x in f.read().strip().splitlines()]

        bboxes = np.array(plot_label, dtype=np.float32)
        labels = bboxes[:, 0]
        # denorm
        bboxes = self.denorm_yolo(bboxes)
        bboxes = bboxes[:, 1:]
        bboxes = self.xywh2xyxy(bboxes)

        img = cv2.imread(img_path)
        # plot and save result
        for box, label in zip(bboxes, labels):
            color = colors(label)
            label = f'{label}'

            # 先画框
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1, (255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)

            if view_img:
                cv2.imshow(str(save_path + img_path), img)
                cv2.waitKey(1)  # 1 millisecond

            if imwrite:
                image_path = img_path.split('/')[-1]
                cv2.imwrite('{}/{}'.format(save_path, image_path), img)
                logging.info(f'save image {image_path} to {save_path}')

    def __call__(self):
        self.compute()
        self.format_result()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='/mnt/data/guozebin/fake_real_data_merge/0302/coco/labels/val/', help='gt txt save path')
    parser.add_argument('--pred_path', type=str, default='/mnt/data/guozebin/object_detection/yolov7/runs/detect/exp/yolov7x_merge/labels/', help='pred result txt save path')
    parser.add_argument('--result_save_path', type=str, default='./yolov7_0302.txt', help='acc result and error and miss result save path')
    args = parser.parse_args()
    logging.info(args)

    #classes_dict = {'红烧大排': 1, '蚝油牛肉': 2, '干锅鸡': 3, '红烧狮子头': 4, '素鸡小烧肉': 5, '蒜蓉肉丝': 6 ,'莴笋炒蛋':7 ,'鱼香茄子':8, '麻婆豆腐': 9, '芹菜百叶司': 10, '老南瓜': 11, '大白菜油豆腐': 12, '冰红茶':13, '老酸奶':14}
    compute = Cumpute_pred_acc(pred_path=args.pred_path, label_path=args.label_path, result_sava_path=args.result_save_path, mask=None)
    compute()

    # # test plot
    # img_path = '/mnt/data/guozebin/object_detection/yolov7/data/coco/images/val/'
    # txt_path = args.pred_path
    # assert len(os.listdir(img_path)) == len(os.listdir(txt_path)), 'When batch drawing, the number of labels and the number of pictures must match!'
    # for img, gt in zip(sorted(os.listdir(img_path)),sorted(os.listdir(txt_path))):
    #     img_paths = os.path.join(img_path, gt.replace('txt','jpg'))
    #     gt_txt_paths = os.path.join(txt_path, gt)
    #     compute.plot_bbox(img_path=img_paths, txt_path=gt_txt_paths, save_path='../visualization/')
