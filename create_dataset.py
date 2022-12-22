import os, cv2

from pathlib import Path
import numpy as np
import json
import sys
sys.path.append("/mnt/data/guozebin/object_detection/")
from statistics_class import Statistics_class

"""
用于将yolo格式的目标检测数据集转换为分类数据集合适，并写了一些策略，比如使用find_max_real_class_numbers函数去计算数据集中每个类的实例个数，
从而制定分类数据集样本平衡方案。
"""

# create index for all classes
def create_index():
    single_dict_path = '/0302_food_dict.json'
    all_dict_path = '/mnt/data/guozebin/object_detection/all_food_dict.json'

    with open(single_dict_path,'r') as f:
        single_dict = json.load(f)

    reverse_single_dict = {k:v for v,k in single_dict.items()}

    with open(all_dict_path,'r') as f:
        all_dict = json.load(f)
    reverse_all_dict = {k:v for v,k in all_dict.items()}

    class_result =[]
    # query
    for k, v in single_dict.items():
        class_result.append(reverse_all_dict[v])
    #class_result.sort(key=lambda x:int(x))
    return class_result

def find_max_real_class_numbers(class_result):
    path = '/mnt/data/guozebin/object_detection/yolov7/data/coco'

    test = Statistics_class(root_path=path, save_path='./')
    dit = test.statistics_from_yolo_txt()
    sort_dit = test.statistics_sort(dit, sort=True)
    max_class_nums = 0
    for key, value in sort_dit['train_set_result'].items():
        if key not in class_result: continue

        if value > max_class_nums:
            max_class_nums = value

    if max_class_nums > 1000:
        max_class_nums = 1000

    return max_class_nums


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# The image size is fixed here. If the size of each image is different, you need to modify the value here
def denorm(box, wh=(1000, 720)):
    bbox = box.copy()
    bbox[:, 0] *= wh[0]
    bbox[:, 1] *= wh[1]
    bbox[:, 2] *= wh[0]
    bbox[:, 3] *= wh[1]

    return bbox

def get_random_data(class_result, max_class_nums, split='train'):
    # 原图片、标签文件、裁剪图片路径
    val_img_path = ['/mnt/data/guozebin/object_detection/yolov7/data/coco/images/val']
    val_txt_path = ['/mnt/data/guozebin/object_detection/yolov7/data/coco/labels/val']

    train_img_path = ['/mnt/data/guozebin/object_detection/yolov7/data/coco/images/train','/mnt/data/guozebin/fake_real_data_mrege/coco/images/train']

    train_txt_path = ['/mnt/data/guozebin/object_detection/yolov7/data/coco/labels/train', '/mnt/data/guozebin/fake_real_data_mrege/coco/labels/train']


    # 声明一个空字典用于储存裁剪图片的类别及其数量
    Numpic = {}
    if split == 'train':
        img_path = train_img_path
        txt_path = train_txt_path
    else:
        img_path = val_img_path
        txt_path = val_txt_path

    obj_img_path = f'/mnt/data/guozebin/object_detection/Secondary_classification/food_fake_real_classification/{split}'

    if not isinstance(img_path, list):
        img_path = [img_path]
    if not isinstance(txt_path, list):
        txt_path = [txt_path]

    for id in range(len(img_path)):
        # 把原图片裁剪后，按类别新建文件夹保存，并在该类别下按顺序编号
        for img_file in os.listdir(img_path[id]):

            print(f'handle img {img_file}')

            if img_file[-4:] in ['.png', '.jpg']:  # 判断文件是否为图片格式
                img_filename = os.path.join(img_path[id], img_file)  # 将图片路径与图片名进行拼接
                img_cv = cv2.imread(img_filename)  # 读取图片
                h, w = img_cv.shape[:2]
                # img_cv = cv2.imread(img_filename)
                # print(img_cv.shape)
                img_name = (os.path.splitext(img_file)[0])  # 分割出图片名，如“000.png” 图片名为“000”
                xml_name = txt_path[id] + '/' + '%s.txt' % img_name  # 利用标签路径、图片名、xml后缀拼接出完整的标签路径名
                if os.path.exists(xml_name):  # 判断与图片同名的标签是否存在，因为图片不一定每张都打标
                    with open(xml_name,'r') as f:
                        lables = [x.split() for x in f.read().strip().splitlines()]

                    lables = np.array(lables, dtype=np.float32)
                    cla = lables[:, 0]

                    lables = xywh2xyxy(lables[:, 1:])
                    lables = denorm(lables,wh=(w,h))
                    for xyxy, cla in zip(lables, cla):  # 遍历所有目标框
                        x0, y0, x1,y1 = xyxy
                        name = str(int(cla))
                        if name not in class_result: continue

                        obj_img = img_cv[int(y0):int(y1), int(x0):int(x1)]  # cv2裁剪出目标框中的图片
                        if 0 in obj_img.shape:
                            continue

                        Numpic.setdefault(name, 0)  # 判断字典中有无当前name对应的类别，无则新建
                        Numpic[name] += 1  # 当前类别对应数量 + 1
                        my_file = Path(obj_img_path + '/' + name)  # 判断当前name对应的类别有无文件夹

                        if 1 - my_file.is_dir():  # 无则新建
                            os.mkdir(obj_img_path + '/' + str(name))

                        if Numpic[name] >= max_class_nums: continue

                        cv2.imwrite(obj_img_path + '/' + name + '/' + '%04d' % (Numpic[name]) + '.jpg',
                                    obj_img)  # 保存裁剪图片，图片命名4位，不足补0

if __name__ == '__main__':
    class_result = create_index()
    max_class_nums = find_max_real_class_numbers(class_result)
    get_random_data(class_result, max_class_nums, split='train')
