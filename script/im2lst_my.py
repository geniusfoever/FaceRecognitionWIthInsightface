# -*- coding: utf-8 -*-
#aligned_images_DB
#--john
#----12.jpg
#----13.jpg

import sys
import os
from easydict import EasyDict as edict
import argparse

if __name__ =="__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    args=parser.add_argument('--root',
                        default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3/train/',
                        help='path to folder containing images.')
    args=parser.add_argument('--output_folder',
                        default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3/train/',
                        help='path to folder containing images.')

    args = parser.parse_args()
input_dir = args.root
fp = open(os.path.join(args.output_folder,'prefix.lst'), 'w')
ret = []
label = 0
person_names = []
for person_name in os.listdir(input_dir):
  person_names.append(person_name)
person_names = sorted(person_names)
for person_name in person_names:
  _subdir = os.path.join(input_dir, person_name)
  if not os.path.isdir(_subdir):
    continue
  #for _subdir2 in os.listdir(_subdir):
    #_subdir2 = os.path.join(_subdir, _subdir2)
    #if not os.path.isdir(_subdir2):
    #  continue
  _ret = []
  for img in os.listdir(_subdir):
      fimage = edict()
      fimage.id = os.path.join(_subdir, img)
      fimage.classname = str(label)
      fimage.image_path = os.path.join(_subdir, img)
      fimage.bbox = None
      fimage.landmark = None
      _ret.append(fimage)
  ret += _ret
  label+=1
for i,item in enumerate(ret):
  if i%1000 == 0: print("%d\t%s\t%d" % (i, item.image_path, int(item.classname)))
  fp.write("%d\t%s\t%d\n" % (1, item.image_path, int(item.classname)))