# -*- coding: utf-8 -*-
# __author__:bin_ze
# 9/5/22 12:11 PM

import json

if __name__ == '__main__':

    single_dict_path = './food_txt/0301_food_dict.json'
    all_dict_path = 'all_food_dict.json'

    with open(single_dict_path, 'r') as f:
        single_dict = json.load(f)

    reverse_single_dict = {k:v for v,k in single_dict.items()}

    with open(all_dict_path,'r') as f:
        all_dict = json.load(f)
    reverse_all_dict = {k:v for v,k in all_dict.items()}


    class_result =[]
    # query
    for k, v in single_dict.items():
        class_result.append(reverse_all_dict[v])
    class_result.sort(key=lambda x:int(x))
    print(class_result)

    hash_map ={reverse_all_dict[v]:reverse_single_dict[v] for v in single_dict.values()}
    print(hash_map)

