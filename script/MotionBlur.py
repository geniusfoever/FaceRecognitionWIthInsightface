import random

import cv2
import numpy as np
import argparse
import copy
import os

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
    default=10,
)
args = parser.parse_args()

def add_motion_blur(walk_args):
    path, dirs, files=walk_args[0]
    img_index_list=[]
    for f in files:
        split_path = f.rsplit(".")
        img_index_list.append(int(split_path[0]))
    # Process files in the directory if any
    for f in files:

        split_path = f.rsplit(".")
        img_index=int(split_path[0])
        if img_index>200000000:
            continue
        elif img_index+200000000 in img_index_list:
            continue
        image_path = path + "/" + f

        #write_path = os.path.join(my_args.outpath, os.path.relpath(path, my_args.path))

        img = cv2.imread(image_path)

        # Specify the kernel size.
        # The greater the size, the more the motion.
        v_kernel_size = random.randrange(1,3)
        h_kernel_size = random.randrange(3,15)
        # Create the vertical kernel.
        kernel_v = np.zeros((v_kernel_size, v_kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.zeros((h_kernel_size, h_kernel_size))

        # Fill the middle row with ones.
        kernel_v[:, int((v_kernel_size - 1) / 2)] = np.ones(v_kernel_size)
        kernel_h[int((h_kernel_size - 1) / 2), :] = np.ones(h_kernel_size)

        # Normalize.
        kernel_v /= v_kernel_size
        kernel_h /= h_kernel_size

        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(img, -1, kernel_v)

        # Apply the horizontal kernel.
        final_mb = cv2.filter2D(vertical_mb, -1, kernel_h)

        # Save the outputs.
        w_path = (
                path
                + "/"
                + str(int(split_path[0])+200000000)

                + "."
                + split_path[1]
        )
        cv2.imwrite(w_path, final_mb)
from multiprocessing import Pool

from itertools import repeat
from functools import partial
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    func=partial(add_motion_blur)
    if args.folder_depth==1:
        with Pool(processes=args.process) as pool:
            results = pool.map(func, zip(os.walk(args.path), repeat((args))))
    else:
        depth=args.folder_depth
        direct_list=[]
        for i in range(depth-1):
            if not direct_list:
                for root, dirs, _ in os.walk(args.path, topdown=False):
                    for name in dirs:
                        direct_list.append(os.path.join(root, name))
            else:
                root_list=copy.deepcopy(direct_list)
                direct_list=[]
                for root in root_list:
                    for _, dirs,_ in os.walk(root):
                        for name in dirs:
                            direct_list.append(os.path.join(root, name))

        for root in direct_list:
            with Pool(processes=args.process) as pool:
                results = pool.map(func, zip(os.walk(root), repeat((args))))


