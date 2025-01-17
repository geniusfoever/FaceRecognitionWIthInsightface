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
# 图片数据文件夹
INPUT_DATA = 'E:/dataset/celeba/classed_celeba'
pairs_file_path = r'E:/dataset/celeba/pairs.txt'

rootdir_list = os.listdir(INPUT_DATA)
idsdir_list = [name for name in rootdir_list if os.path.isdir(os.path.join(INPUT_DATA, name))]

id_nums = len(idsdir_list)
mask_probability=0.8
blur_probaility=0.3
brightness_contrast_probability=0.2
def produce_same_pairs():
    matched_result = []  # 不同类的匹配对
    while(len(matched_result)<24000):
        id_int = random.randint(0, id_nums - 1)

        id_dir = os.path.join(INPUT_DATA, idsdir_list[id_int])


        id_imgs_list = os.listdir(id_dir)

        id_list_len = len(id_imgs_list)

        if id_list_len<1: continue
        for _ in range(10):
            id1_img_file = id_imgs_list[random.randint(0, id_list_len - 1)]
            id2_img_file = id_imgs_list[random.randint(0, id_list_len - 1)]
            if id1_img_file!=id2_img_file: break
        else: continue

        id1_img_file=os.path.join(id_dir,id1_img_file)
        id2_img_file=os.path.join(id_dir,id2_img_file)
        assert os.path.isfile(id1_img_file)

        assert os.path.isfile(id2_img_file)
        same = 1
        #print([id1_path + '\t' + id2_path + '\t',same])
        matched_result.append((id1_img_file + '\t' + id2_img_file + '\t',same))

    return matched_result


def produce_unsame_pairs():
    unmatched_result = []  # 不同类的匹配对


    while(len(unmatched_result)<24000):
        id1_int = random.randint(0, id_nums - 1)
        id2_int = random.randint(0, id_nums - 1)
        while id1_int == id2_int:
            id1_int = random.randint(0, id_nums - 1)
            id2_int = random.randint(0, id_nums - 1)

        id1_dir = os.path.join(INPUT_DATA,  idsdir_list[id1_int])
        id2_dir = os.path.join(INPUT_DATA,  idsdir_list[id2_int])

        id1_imgs_list = os.listdir(id1_dir)
        id2_imgs_list = os.listdir(id2_dir)




        id1_list_len = len(id1_imgs_list)
        id2_list_len = len(id2_imgs_list)
        if id1_list_len==0 or id2_list_len==0: continue
        id1_img_file = id1_imgs_list[random.randint(0, id1_list_len - 1) if id1_list_len>1 else 0]
        id2_img_file = id2_imgs_list[random.randint(0, id2_list_len - 1) if id2_list_len>1 else 0]


        id1_img_file=os.path.join(id1_dir,id1_img_file)
        id2_img_file=os.path.join(id2_dir,id2_img_file)
        assert os.path.isfile(id1_img_file)
        assert os.path.isfile(id2_img_file)
        same = 0
        unmatched_result.append((id1_img_file + '\t' + id2_img_file + '\t', same))
    return unmatched_result

if __name__=="__main__":
    same_result = produce_same_pairs()
    unsame_result = produce_unsame_pairs()

    all_result = same_result + unsame_result

    random.shuffle(all_result)
    #print(all_result)
    print(len(all_result))

    file = open(pairs_file_path, 'w')
    for line in all_result:
        file.write(line[0] + str(line[1]) + '\n')

    file.close()

