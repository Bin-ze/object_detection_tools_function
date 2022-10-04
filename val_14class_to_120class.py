
import os
from tqdm import tqdm

import json
DAY = '0301'
single_dict_path = f'food_txt/{DAY}_food_dict.json'
all_dict_path = 'all_food_dict.json'

with open(single_dict_path,'r') as f:
    single_dict = json.load(f)

reverse_single_dict = {k:v for v,k in single_dict.items()}

with open(all_dict_path,'r') as f:
    all_dict = json.load(f)
reverse_all_dict = {k:v for v,k in all_dict.items()}


hash_map ={reverse_all_dict[v]:reverse_single_dict[v] for v in single_dict.values()}

reverse_hash = {k: v for v, k in hash_map.items()}

def mapping(txt_path):
        # read
        with open(txt_path, 'r') as f:
              res  = [x.split() for x in f.read().strip().splitlines()]
        # change
        for line in res:
                if line[0] in reverse_hash:
                        line[0] = reverse_hash[line[0]]

        # rewrite
        if os.path.exists(txt_path):
                os.remove(txt_path)
        for (cls, *xywh) in res:
                xywh = [float(i) for i in xywh]
                line = (int(cls),)+ tuple(xywh)
                with open(txt_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
        return



if __name__ == "__main__":
        split_list = ['train', 'val', 'test']
        for split in split_list:
                root_path = f'/mnt/data/guozebin/fake_real_data_merge/{DAY}/coco/labels/{split}'
                for txt_file in tqdm(os.listdir(root_path)):
                        txt_path = os.path.join(root_path, txt_file)
                        #mapping(txt_path)