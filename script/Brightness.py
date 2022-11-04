import random

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
    default=r"D:\DataBase\51\lfw_output",
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
    default=16,
)
args = parser.parse_args()
def controller(img, brightness=255,
               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

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
        if img_index>400000000:
            continue
        elif img_index+400000000 in img_index_list:
            pass
        image_path = path + "/" + f

        #write_path = os.path.join(my_args.outpath, os.path.relpath(path, my_args.path))

        img = cv2.imread(image_path)
        brightness=random.randrange(175,340)
        contrast=random.randrange(75,190)
        img=controller(img,brightness,contrast)
        # Save the outputs.
        w_path = (
                path
                + "/"
                + str(int(split_path[0])+400000000)

                + "."
                + split_path[1]
        )
        cv2.imwrite(w_path, img)
from multiprocessing import Pool

from itertools import repeat
from functools import partial
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
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
                pool.map(func, zip(os.walk(root), repeat((args))))


