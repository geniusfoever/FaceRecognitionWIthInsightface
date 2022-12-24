import random
import sys

import cv2
import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default=r"E:\dataset\glint",
    help="Path to either the folder containing images or the image itself",
)
parser.add_argument(
    "--folder_depth",
    type=int,
    default=2,
)
parser.add_argument(
    "--process",
    type=int,
    default=12,
)
args = parser.parse_args()
def controller(img, brightness=0.5,
               contrast=1.):

    cal = cv2.addWeighted(img, brightness,
                          img, 0, 1-brightness)









    #
    # lab = cv2.cvtColor(cal, cv2.COLOR_BGR2LAB)
    # l_channel, a, b = cv2.split(lab)
    #
    # # Applying CLAHE to L-channel
    # # feel free to try different values for the limit and grid size:
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(1, 1))
    # cl = clahe.apply(l_channel)
    #
    # # merge the CLAHE enhanced L-channel with the a and b channel
    # limg = cv2.merge((cl, a, b))
    #
    # # Converting image from LAB Color model to BGR color spcae
    # enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cal=cv2.addWeighted(cal,0,enhanced_img,1,0)
    # Stacking the original image with the enhanced image


    # The function addWeighted calculates
    # the weighted sum of two arrays
    # cal = cv2.addWeighted(img, brightness,
    #                       img, 0, 1-brightness)
    #
    # cal = cv2.addWeighted(cal, contrast, cal, 0, Gamma)
    #
    # cal = cv2.addWeighted(cal, Alpha,
    #                       cal, 0, Gamma)
    # #
    # if contrast != 0:
    #     Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    #     Gamma = 127 * (1 - Alpha)
    #
    #     # The function addWeighted calculates
    #     # the weighted sum of two arrays

    return cal
def change_contrast_brightness(walk_args):
    path, dirs, files=walk_args[0]
    img_index_list=[]
    for f in files:
        split_path = f.rsplit(".")
        img_index_list.append(int(split_path[0]))
    # Process files in the directory if any
    for f in files:

        split_path = f.rsplit(".")
        img_index=int(split_path[0])
        if img_index<400000000:
            continue

        image_path = path + "/" + f

        #write_path = os.path.join(my_args.outpath, os.path.relpath(path, my_args.path))

        img = cv2.imread(image_path)
        brightness=random.random()/2+0.3
        contrast=random.random()*2
        img=controller(img,brightness,contrast)
        # Save the outputs.
        # cv2.imshow(f,img)
        # cv2.waitKey(0)
        cv2.imwrite(image_path, img)
        # print(image_path)



from multiprocessing import Pool

from itertools import repeat
from functools import partial
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=
    # True):
    #     add_mask(walk,args)
    #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    func=partial(change_contrast_brightness)
    if args.folder_depth==1:
        with Pool(processes=args.process) as pool:
            results =list(tqdm( pool.map(func, zip(os.walk(args.path), repeat((args))))))
    else:
        depth=args.folder_depth
        direct_list=[]
        for i in range(depth-1):
            if not direct_list:
                for name in next(os.walk(args.path))[1]:
                    print(name)
                    direct_list.append(os.path.join(args.path, name))
            else:
                root_list=direct_list
                direct_list=[]
                for root in root_list:
                    print(name)
                    for name in next(os.walk(root))[1]:
                        direct_list.append(os.path.join(root, name))
        for root in direct_list:
            with Pool(processes=args.process) as pool:
                list(tqdm(pool.map(func, zip(os.walk(root), repeat((args)))), total=73382*10))



