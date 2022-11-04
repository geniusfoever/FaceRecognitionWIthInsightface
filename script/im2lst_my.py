# -*- coding: utf-8 -*-
#aligned_images_DB
#--john
#----12.jpg
#----13.jpg
import random
import sys
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
brightness_contrast_probability=0.2
blur_probaility=0.3
mask_probability=0.5
if __name__ =="__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    args=parser.add_argument('--root',
                        default=r'D:\DataBase\51\lfw_output\train',
                        help='path to folder containing images.')
    args=parser.add_argument('--output_folder',
                        default=r'D:\DataBase\51\lfw_output',
                        help='path to folder containing images.')

    args = parser.parse_args()
input_dir = args.root
fp = open(os.path.join(args.output_folder,'prefix.lst'), 'w')
label = 0
person_names = []
ret=[]
index=0
for person_name in os.listdir(input_dir):
  person_names.append(person_name)
for person_name in tqdm(person_names):
  _subdir = os.path.join(input_dir, person_name)
  if not os.path.isdir(_subdir):
    continue
  #for _subdir2 in os.listdir(_subdir):
    #_subdir2 = os.path.join(_subdir, _subdir2)
    #if not os.path.isdir(_subdir2):
    #  continue
  _ret = []
  img_index_list=[]
  for img in os.listdir(_subdir):
      img_index_list.append(int(img.split('.')[0]))
  for img_index in img_index_list:
      if img_index>100000000:continue
      for _ in range(10):
          a = img_index + ((100000000 if random.random() < mask_probability else 0) +
                              (200000000 if random.random() < blur_probaility else 0) +
                              (400000000 if random.random() < brightness_contrast_probability else 0))
          if a in img_index_list:
              img_index = a
              break
      fimage = edict()
      fimage.id = img_index
      fimage.classname = str(label)
      img_name=str(img_index)
      img_name = img_name if len(img_name) >= 4 else (4 - len(img_name)) * '0' + img_name
      fimage.image_path = person_name+'/'+ img_name+".jpg"
      index+=1
      fimage.index=index
      # fimage.bbox = None
      # fimage.landmark = None
      _ret.append(fimage)
  ret+=_ret
  label+=1
random.shuffle(ret)
for item in ret:
      fp.write("%d\t%d\t%s\n" % (item.index, int(item.classname), item.image_path))