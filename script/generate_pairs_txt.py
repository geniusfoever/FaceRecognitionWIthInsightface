# coding:utf-8
import glob
import os.path
import numpy as np
import os
import re
'''
创建验证集bin的pairs.txt
'''
import random
from tqdm import tqdm
# import argparse
# # 图片数据文件夹
# parser = argparse.ArgumentParser(description='do dataset merge')
# # general
# parser.add_argument('--input', default=4, type=int)
# parser.add_argument('--output', default=r"G:\My Drive\insightface\dataset\glink_360m\image", type=str, help='')
# args = parser.parse_args()
INPUT_DATA = 'D:/database/51/lfw_output/test'
pairs_file_path = 'D:/database/51/lfw_output/pairs2.txt'

the_rootdir_list = os.listdir(INPUT_DATA)
idsdir_list = [name for name in the_rootdir_list if os.path.isdir(os.path.join(INPUT_DATA, name))]

id_nums = len(idsdir_list)

mask_probability=1
blur_probaility=0.3
brightness_contrast_probability=0.2
def produce_same_pairs(dir_name):
    matched_result = []  # 不同类的匹配对

    id_dir = os.path.join(INPUT_DATA, dir_name)

    id_imgs_list = os.listdir(id_dir)


    imgs_index_list=[]
    for img_name in id_imgs_list:
        imgs_index_list.append(int(img_name.split('.')[0]))
    id_list_len = len(id_imgs_list)
    if sum(i < 100000000 for i in imgs_index_list) <= 1: return None

    for id1_img_file in imgs_index_list:
        if id1_img_file>100000000:continue
        id2_img_file = imgs_index_list[random.randint(0, id_list_len - 1)] % 100000000
        while id1_img_file == id2_img_file:
            id2_img_file = imgs_index_list[random.randint(0, id_list_len - 1)] % 100000000

        for _ in range(10):
            a = id2_img_file + ((100000000 if random.random() < mask_probability else 0) +
                                (200000000 if random.random() < blur_probaility else 0) +
                                (400000000 if random.random() < brightness_contrast_probability else 0))
            if a in imgs_index_list:
                id2_img_file = a
                break

        id1_name = str(id1_img_file)
        id2_name = str(id2_img_file)
        id1_name = id1_name if len(id1_name) >= 4 else (4 - len(id1_name)) * '0' + id1_name
        id2_name = id2_name if len(id2_name) >= 4 else (4 - len(id2_name)) * '0' + id2_name
        id1_path = os.path.join(id_dir, id1_name + ".jpg")
        id2_path = os.path.join(id_dir, id2_name + ".jpg")

        assert os.path.isfile(id1_path)
        assert os.path.isfile(id2_path)
        same = 1
        # print([id1_path + '\t' + id2_path + '\t',same])
        matched_result.append((id1_path + '\t' + id2_path + '\t', same))

    return matched_result


def produce_unsame_pairs(dir_name12):
    unmatched_result = []  # 不同类的匹配对
    dir_name1=dir_name12[0]
    dir_name2=dir_name12[1]
    if dir_name2==dir_name1: return None



    id1_dir = os.path.join(INPUT_DATA, dir_name1)
    id2_dir = os.path.join(INPUT_DATA, dir_name2)

    id1_imgs_list = os.listdir(id1_dir)
    id2_imgs_list = os.listdir(id2_dir)
    id1_list_len = len(id1_imgs_list)
    id2_list_len = len(id2_imgs_list)

    imgs_index_list = []
    for img_name in id1_imgs_list:
        imgs_index_list.append(int(img_name.split('.')[0]))

    if id1_list_len == 0 or id2_list_len == 0: return None

    for id2_img_file in id2_imgs_list:
        if int(id2_img_file.split('.')[0])>100000000: continue
        id1_img_file = imgs_index_list[random.randint(0, id1_list_len - 1) if id1_list_len > 1 else 0] % 100000000
        for _ in range(10):
            a = id1_img_file + ((100000000 if random.random() < mask_probability else 0) +
                                (200000000 if random.random() < blur_probaility else 0) +
                                (400000000 if random.random() < brightness_contrast_probability else 0))
            if a in imgs_index_list:
                id1_img_file = a
                break
        id1_name = str(id1_img_file)
        id2_name = str(int(id2_img_file.split('.')[0]) % 100000000)
        id1_name = id1_name if len(id1_name) >= 4 else (4 - len(id1_name)) * '0' + id1_name
        id2_name = id2_name if len(id2_name) >= 4 else (4 - len(id2_name)) * '0' + id2_name
        id1_path = os.path.join(id1_dir, id1_name + ".jpg")
        id2_path = os.path.join(id2_dir, id2_name + ".jpg")
        assert os.path.isfile(id1_path)
        assert os.path.isfile(id2_path)
        same = 0
        unmatched_result.append((id1_path + '\t' + id2_path + '\t', same))

    return unmatched_result
from multiprocessing import Pool

from itertools import chain
from functools import partial
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    func=partial(produce_same_pairs)
    with Pool(processes=32) as pool:
        same_results_list= list(tqdm(pool.map(func, the_rootdir_list),total=len(the_rootdir_list)))

    same_result = list(chain.from_iterable([i for i in same_results_list if i is not None]))
    dir_part_list=[]
    for dir1 in the_rootdir_list:
        for dir2 in the_rootdir_list:
            dir_part_list.append((dir1,dir2))
    func = partial(produce_unsame_pairs)
    with Pool(processes=32) as pool:
        unsame_results_list = list(tqdm(pool.map(func, dir_part_list),total=len(dir_part_list)))
    unsame_result =list(chain.from_iterable([i for i in unsame_results_list if i is not None]))

    all_result = same_result + unsame_result

    random.shuffle(all_result)
    #print(all_result)

    file = open(pairs_file_path, 'w')
    for line in all_result:
        file.write(line[0] + str(line[1]) + '\n')

    file.close()

