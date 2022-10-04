# -*- coding: utf-8 -*-
# __author__:bin_ze
# 8/14/22 9:55 AM
import json
import os
import numpy as np

class Trans_labels:
    def __init__(self, root_path, save_path='./data/coco_kpts/'):
        self.root = root_path
        self.save_path = save_path

        self.train_txt = save_path + '/train2017.txt'
        self.val_txt = save_path + '/val2017.txt'

        if os.path.exists(self.train_txt):
            os.remove(self.train_txt)
        if os.path.exists(self.val_txt):
            os.remove(self.val_txt)

        self.classes = {
            'lying':0,
            'standing':1,
            'sitting':2,
            'squatting':3,
            'others':4,
            'lying_unsure':5,
            'standing_unsure':6,
            'sitting_unsure':7,
            'squatting_unsure':8,
            'others_unsure':9
        }

    def rw_json(self, json_path, taken):
        '''

        Args:
            json_path:

        Returns:

        '''
        with open(json_path, 'r') as f:
            label = json.load(f)


        if taken % 10 != 0:
            write_path = self.save_path + 'labels/train2017/' + json_path.split('/')[-1].replace('json','txt')
            with open(self.train_txt, 'a') as f:
                f.write(json_path.replace('json','jpg') + '\n')

        else:
            write_path = self.save_path + 'labels/val2017/' + json_path.split('/')[-1].replace('json', 'txt')

            with open(self.val_txt, 'a') as f:
                f.write(json_path.replace('json','jpg') + '\n')

        for instance in label:

            w, h  = instance['width'], instance['height']

            # label format
            bbox =  instance['bbox']
            bbox = self.norm_bbox(bbox, w, h)

            # class
            cls = self.classes[instance['box_name']]
            #cls = 0

            # keypoints
            keypoints = instance['keypoints']
            keypoints = self.norm_keypoints(keypoints, w, h)

            line = (cls, *bbox, *keypoints)

            # write
            with open(write_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


    def __call__(self):
        '''

        Returns:

        '''
        train_taken = 0
        print(f'total data: {len(os.listdir(self.root))//2}')
        for path in os.listdir(self.root):
            if path.endswith('json'):
                json_path = os.path.join(self.root, path)
                self.rw_json(json_path, taken=train_taken)
                train_taken += 1


    def norm_bbox(self, bbox, w, h):
        '''

        Args:
            bbox: list [x,y,w,h]

        Returns:

        '''
        box = bbox.copy()

        box[0] = (box[0] + 0.5 * box[2])/ w
        box[1] = (box[1] + 0.5 * box[3])/ h
        box[2] = box[2] / w
        box[3] = box[3] / h

        return box

    def norm_keypoints(self, keypoints, w, h):
        '''

        Args:
            keypoints:
            w:
            h:

        Returns:

        '''
        keypoint = keypoints.copy()

        keypoint[::3] = np.array(keypoint[::3]) / w
        keypoint[1::3] = np.array(keypoint[1::3]) / h

        return keypoint


tran = Trans_labels(root_path='/mnt/data/guozebin/pose/1262')
tran()