import os.path as osp
import xml.etree.ElementTree as ET

import json
import os
import logging
import argparse
import sys

from glob import glob
from tqdm import tqdm
from PIL import Image

'''
xml format to json format 
'''
class Voc_to_coco:

    def __init__(self, xml_path, img_path, out_file):
        """
        :param xml_path: voc 标注目录
        :param img_path: voc 图片目录
        :param out_file: json 文件保存路径
        """
        self.xml_path = xml_path
        self.img_path = img_path
        self.out_file = out_file
        self.label_ids = {name: i + 1 for i, name in enumerate(self.object_classes())}
        logging.info(self.label_ids)

    @staticmethod
    def object_classes():
        """
        :return: 静态方法，手动填入标注数据集的类别
        """
        return ['car']

    @staticmethod
    def get_segmentation(points):
        """
        :param points: [xmin, ymin ,w, h]
        :return: bbox segmentation
        """
        return [points[0], points[1], points[2] + points[0], points[1],
                 points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]

    def parse_xml(self, xml_path, img_id, anno_id):
        """
        :param xml_path:
        :param img_id:
        :param anno_id:
        :return: single image coco annotations
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotation = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.object_classes(): #当不在自己所写的类别内时
                continue
            category_id = self.label_ids[name]
            bnd_box = obj.find('bndbox')
            xmin = int(bnd_box.find('xmin').text)
            ymin = int(bnd_box.find('ymin').text)
            xmax = int(bnd_box.find('xmax').text)
            ymax = int(bnd_box.find('ymax').text)
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            area = w*h
            segmentation = self.get_segmentation([xmin, ymin, w, h])
            annotation.append({
                            "segmentation": segmentation,
                            "area": area,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": [xmin, ymin, w, h],
                            "category_id": category_id,
                            "id": anno_id,
                            "ignore": 0})
            anno_id += 1
        return annotation, anno_id

    def cvt_annotations(self, img_path, xml_path, out_file):
        """
        :param img_path:
        :param xml_path:
        :param out_file:
        :return: finally annotations
        """
        images = []  # img annotations
        annotations = []

        img_id = 1
        anno_id = 1
        logging.info('start trans')
        # glob 支持通配符并返回可迭代对象
        for img_path in tqdm(glob(img_path + '/*.jpg')):
            w, h = Image.open(img_path).size
            img_name = osp.basename(img_path)
            img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
            images.append(img)

            xml_file_name = img_name.split('.')[0] + '.xml'
            xml_file_path = osp.join(xml_path, xml_file_name)
            annos, anno_id = self.parse_xml(xml_file_path, img_id, anno_id)
            annotations.extend(annos)
            img_id += 1

        categories = []
        for k, v in self.label_ids.items():
            categories.append({"name": k, "id": v})
        final_result = {"images": images, "annotations": annotations, "categories": categories}
        with open(out_file, 'w') as w:
            w.write(json.dumps(final_result))
        return annotations

    def __call__(self):
        """
        __call__用于类直接调用自身
        :return:
        """
        self.cvt_annotations(self.img_path, self.xml_path, self.out_file)
        logging.info('trans Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/data/VOCdevkit/VOC2007/JPEGImages', help='img file path')
    parser.add_argument('--xml_path', type=str, default='/data/VOCdevkit/VOC2007/Annotations', help='xml file path')
    parser.add_argument('--out_file', type=str, default='./annotations.json', help='coco json outfile path')
    args = parser.parse_args()
    logging.info(args)
    if not os.path.exists(args.out_file):
        f = open(args.out_file, 'w')
        f.close()

    # 实例化
    trans = Voc_to_coco(xml_path=args.xml_path, img_path=args.img_path, out_file=args.out_file)
    # call
    trans()