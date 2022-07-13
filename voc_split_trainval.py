import os
import random
import sys
import argparse
import logging

class Voc_split_trainval:
    def __init__(self, root_path, txtsavepath):
        self.root_path = root_path
        self.xmlfilepath = self.root_path + '/Annotations'
        self.txtsavepath = txtsavepath

    def split_trainval(self, trainval_percent=1, train_percent=0.9):

        if not os.path.exists(self.txtsavepath):
            os.makedirs(self.txtsavepath)

        trainval_percent = trainval_percent
        train_percent = train_percent
        total_xml = os.listdir(self.xmlfilepath)
        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        logging.info(f"train and val size:{tv}")
        logging.info(f"train size:{tr}")

        ftrainval = open(self.txtsavepath + '/trainval.txt', 'w')
        ftest = open(self.txtsavepath + '/test.txt', 'w')
        ftrain = open(self.txtsavepath + '/train.txt', 'w')
        fval = open(self.txtsavepath + '/val.txt', 'w')

        for i in list:
            if total_xml[i][-4:] != '.xml': continue
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
    def __call__(self):
        self.split_trainval(trainval_percent=1, train_percent=0.9)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='', help='dataset path')
    parser.add_argument('--txt_save_path', type=str, default='./', help=' dataset path')
    args = parser.parse_args()
    logging.info(args)

    # run statistics function
    test = Voc_split_trainval(root_path=args.root_path,txtsavepath=args.txt_save_path)
    test()