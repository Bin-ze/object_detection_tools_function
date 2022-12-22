import os
import argparse

class Handle_yolo:
    """
    该类用于生成yolo所需要的train.txt, val.txt，test.txt文件
    """
    def __init__(self, root_path):
        self.path = root_path

    def add_img_txt_path(self):

        for split in os.listdir(os.path.join(self.path, 'images')):
            for file in os.listdir(os.path.join(self.path, 'images', split)):
                #if file.split('_')[0] == '0000':continue
                with open(f'{self.path}/{split}.txt', 'a') as f:
                    file_name = os.path.join(self.path, 'images', split, file)
                    f.write(file_name + '\n')

    def __call__(self):
        self.add_img_txt_path()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/mnt/data/guozebin/fake_real_data_0-14/0304/coco/',
                        help='dataset root path')
    args = parser.parse_args()
    handle = Handle_yolo(root_path=args.root_path)
    handle()
