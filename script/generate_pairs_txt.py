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
# 图片数据文件夹
INPUT_DATA = 'D:/database/51/lfw_masked/test'
pairs_file_path = 'D:/database/51/lfw_masked/pairs.txt'

rootdir_list = os.listdir(INPUT_DATA)
idsdir_list = [name for name in rootdir_list if os.path.isdir(os.path.join(INPUT_DATA, name))]

id_nums = len(idsdir_list)

def produce_same_pairs():
    matched_result = []  # 不同类的匹配对
    for dir_name in rootdir_list:
        id_int= random.randint(0,id_nums-1)

        id_dir = os.path.join(INPUT_DATA, dir_name)

        id_imgs_list = os.listdir(id_dir)

        id_list_len = len(id_imgs_list)
        if id_list_len<=1: continue

        for img_id1 in range(id_list_len-1):
            for img_id2 in range(img_id1+1,id_list_len):
                id1_img_file = id_imgs_list[img_id1]

                id2_img_file = id_imgs_list[img_id2]

                id1_path = os.path.join(id_dir, id1_img_file)
                id2_path = os.path.join(id_dir, id2_img_file)

                same = 1
                #print([id1_path + '\t' + id2_path + '\t',same])
                matched_result.append((id1_path + '\t' + id2_path + '\t',same))

    return matched_result


def produce_unsame_pairs():
    unmatched_result = []  # 不同类的匹配对

    for dir_name1 in rootdir_list:
        for dir_name2 in rootdir_list:
            if dir_name2==dir_name1: continue

            id1_int = random.randint(0,id_nums-1)
            id2_int = random.randint(0,id_nums-1)
            while id1_int == id2_int:
                id1_int = random.randint(0,id_nums-1)
                id2_int = random.randint(0,id_nums-1)

            id1_dir = os.path.join(INPUT_DATA, dir_name1)
            id2_dir = os.path.join(INPUT_DATA, dir_name2)

            id1_imgs_list = os.listdir(id1_dir)
            id2_imgs_list = os.listdir(id2_dir)
            id1_list_len = len(id1_imgs_list)
            id2_list_len = len(id2_imgs_list)


            for img_id1 in range(id1_list_len):
                for img_id2 in range(id2_list_len):
                    id1_img_file = id1_imgs_list[img_id1]

                    id2_img_file = id2_imgs_list[img_id2]

                    id1_path = os.path.join(id1_dir, id1_img_file)
                    id2_path = os.path.join(id2_dir, id2_img_file)

                    same = 0
                    unmatched_result.append((id1_path + '\t' + id2_path + '\t',same))
    return unmatched_result


same_result = produce_same_pairs()
unsame_result = produce_unsame_pairs()

all_result = same_result + unsame_result

random.shuffle(all_result)
#print(all_result)

file = open(pairs_file_path, 'w')
for line in all_result:
    file.write(line[0] + str(line[1]) + '\n')

file.close()

