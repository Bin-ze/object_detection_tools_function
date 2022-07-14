import os
import json
import logging
import argparse
import sys

import os.path as osp
import xml.etree.ElementTree as ET

from pycocotools.coco import COCO

'''
Statistics_food_class用于统计目标检测的三种格式：
VOC:xml
COCO:json
YOLO:txt
下的类别数量

'''

class Statistics_class:
    def __init__(self, root_path, save_path):
        self.path = root_path
        self.save_path = save_path
        self.split = ['train', 'val', 'test']
        self.statistics_dict = dict()
        # self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    # yolo format
    def statistics_from_yolo_txt(self):
        """
        :return: 类别数量统计字典(无序)
        """

        logging.info('start statistics instance number')
        train = []
        val = []
        test = []

        for split in os.listdir(osp.join(self.path, 'labels')):
            assert split in self.split,"yolo dataset split not in ['train', 'val', 'test']"

            for txt_file in os.listdir(osp.join(self.path, 'labels', split)):
                file_index = osp.join(self.path, 'labels', split, txt_file)
                with open(file_index, 'r') as f:
                    labels_list = f.readlines()
                for img_label in labels_list:
                    if split == 'train':
                        train.append(img_label.split(' ')[0])
                    elif split == 'val':
                        val.append(img_label.split(' ')[0])
                    else:
                        test.append(img_label.split(' ')[0])
        # statistics
        train_set_cla = dict()
        val_set_cla = dict()
        test_set_cla = dict()

        total_set = train + test + val
        classes = set(total_set)
        for cla in classes:
            train_set_cla.update({cla:train.count(cla)})
            val_set_cla.update({cla:val.count(cla)})
            test_set_cla.update({cla:test.count(cla)})

        self.statistics_dict = dict(
            train_set_result=train_set_cla,
            val_set_result=val_set_cla,
            test_set_result=test_set_cla
        )
        return self.statistics_dict

    # voc format
    def statistics_from_voc_xml(self):
        """
        :return: 类别数量统计字典(无序)
        """

        logging.info('start statistics instance number')
        xml_path = self.path
        classes = []
        all_class_list = []
        for xml in os.listdir(xml_path):
            xml_file = os.path.join(xml_path, xml)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                all_class_list.append(name)
                if name in classes:
                    continue
                else:
                    classes.append(name)
        statistics_ = dict()
        for cla in classes:
            statistics_.update({cla:all_class_list.count(cla)})

        self.statistics_dict = dict(xml_statistics_result=statistics_)
        return self.statistics_dict

    # coco format
    def statistics_from_coco_json(self):
        """
        :return: 类别数量统计字典(无序)
        """
        logging.info('start statistics instance number')
        coco = COCO(self.path)
        classes = {}
        for cla in coco.dataset['categories']:
            classes.update({cla['id']:cla['name']})
        all_class_list = []
        for ann in coco.dataset['annotations']:
            all_class_list.append(ann['category_id'])
        statistics_ = dict()
        for cla in classes.keys():
            statistics_.update({classes[cla]:all_class_list.count(cla)})

        self.statistics_dict = dict(json_statistics_result=statistics_)
        return self.statistics_dict

    # result sort
    def statistics_sort(self, dict_input, sort=False):
        """
        :param dict_input: 统计输出
        :param sort: 排序后的结果输出
        :return:
        """
        logging.info('statistics result sort')
        result_key = list(dict_input.keys())
        for result_key in result_key:
            sort_dict = dict_input[result_key]
            sort_statistics_ = dict()
            for key,val in sorted(sort_dict.items(), key=lambda x:x[1]):
                sort_statistics_.update({key:val})
            self.statistics_dict.update({result_key:sort_statistics_})
        return self.statistics_dict if sort == True else dict_input

    def json_output(self, dict_input):
        logging.info('json formatted output')
        if os.path.isfile(self.save_path):os.remove(self.save_path)
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(dict_input, indent=4))
        logging.info(f'statistics result save in {self.save_path}')
        return

    def __call__(self):
        statistics_dict = self.statistics_from_yolo_txt()
        self.json_output(statistics_dict)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='yolov5 format dataset path or xml or json path')
    parser.add_argument('--save_path', type=str, default='./statistics_result.txt', help='statistics result save path')
    args = parser.parse_args()
    logging.info(args)

    # run statistics function
    test = Statistics_class(root_path=args.dataset_path, save_path=args.save_path)
    dit = test.statistics_from_yolo_txt()
    sort_dit = test.statistics_sort(dit, sort=True)
    test.json_output(sort_dit)


