from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

from tqdm import tqdm

# import facenet
import detect_face
import random
from time import sleep
from insightface.app import FaceAnalysis
#
print(os.path.join(Path(__file__).parent.parent.absolute(), 'common'))
sys.path.append(os.path.join(Path(__file__).parent.parent.absolute(), 'common'))
# print(os.path.join(os.path.abspath(__file__), '..', 'common'))
import face_preprocess as face_preprocess
from skimage import transform as trans
import cv2


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe, GTframe):
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2] - Reframe[0];
    height1 = Reframe[3] - Reframe[1];

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio
# def list_image(root, recursive, exts):
#     """Traverses the root of directory that contains images and
#     generates image list iterator.
#     Parameters
#     ----------
#     root: string
#     recursive: bool
#     exts: string
#     Returns
#     -------
#     image iterator that contains all the image under the specified path
#     """
#
#     i = 0
#     if recursive:
#         cat = {}
#         for path, dirs, files in os.walk(root, followlinks=True):
#             dirs.sort()
#             files.sort()
#             for fname in files:
#                 fpath = os.path.join(path, fname)
#                 suffix = os.path.splitext(fname)[1].lower()
#                 if os.path.isfile(fpath) and (suffix in exts):
#                     if path not in cat:
#                         cat[path] = len(cat)
#                     yield (i, os.path.relpath(fpath, root), cat[path])
#                     i += 1
#         for k, v in sorted(cat.items(), key=lambda x: x[1]):
#             print(os.path.relpath(k, root), v)
#     else:
#         for fname in sorted(os.listdir(root)):
#             fpath = os.path.join(root, fname)
#             suffix = os.path.splitext(fname)[1].lower()
#             if os.path.isfile(fpath) and (suffix in exts):
#                 yield (i, os.path.relpath(fpath, root), 0)
#                 i += 1

def main(args,start,end,p_id):

    print('Creating networks and loading parameters')

    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    minsize = 60
    threshold = [0.6, 0.85, 0.8]
    factor = 0.85

    # Add a random key to the filename to allow alignment using multiple processes
    # random_key = np.random.randint(0, high=99999)
    # bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    # output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    output_filename = os.path.join(args.outdir, 'lst')

    nrof_images_total = 0
    nrof = np.zeros((5,), dtype=np.int32)
    face_count = 0
    t=tqdm(range(start,end),position=p_id)
    for folder in t:

        for img_name in os.listdir(os.path.join(args.indir,str(folder))):
            src_img_path = os.path.join(args.indir,str(folder),img_name)
            # if nrof_images_total % 100 == 0:
            #     print("Processing %d, (%s)" % (nrof_images_total, nrof))
            # nrof_images_total += 1
            if not os.path.exists(src_img_path):
                print('image not found (%s)' % src_img_path)
                continue
            # print(image_path)
            try:
                img = cv2.imread(src_img_path)

            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(src_img_path, e)
                print(errorMessage)
            else:
                if img.ndim < 2:
                    print('Unable to align "%s", img dim error' % src_img_path)
                    # text_file.write('%s\n' % (output_filename))
                    continue
                if img.ndim == 2:
                    img = to_rgb(img)
                img = img[:, :, 0:3]


                target_dir = os.path.join(args.outdir, str(folder))
                print(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir,exist_ok=True)

                _minsize = minsize
                _bbox = None
                _landmark = None

                width=img.shape[1]
                height=img.shape[0]
                if face_count==0:
                    print("Width: {} Height: {}".format(width,height))
                faces = app.get(img)
                if len(faces)<1:
                    continue
                if len(faces)>1:

                    min_distance_to_centre = 10000000000
                    face=None
                    for f in faces:
                        bbox = f.bbox
                        distance = (bbox[0] + 0.5 * bbox[2] - 0.5*width) ** 2 + (bbox[1] + 0.5 * bbox[3] - 0.5*height) ** 2- \
                                   0.1*(bbox[2]**2+bbox[3]**2)
                        print(distance)
                        if distance < min_distance_to_centre:
                            min_distance_to_centre = distance
                            face=f
                else: face=faces[0]
                if not face:
                    print("Face is None ", str(folder), img_name,len(faces),faces)
                    continue
                bounding_boxe=face.bbox
                points=face.kps

                #print(points)
                if points == []:
                    print(src_img_path)
                else:
                    _landmark = points.T

                #print(_landmark.shape)
                warped = face_preprocess.preprocess(img, bbox=bounding_boxe, landmark=_landmark.reshape([2,5]).T, image_size=args.image_size)

                #cv2.imshow(str(num),bgr)
                target_file = os.path.join(args.outdir,str(folder),img_name)
                face_count+=1
                print(target_file)
                cv2.imwrite(target_file,warped)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir',
                        default='E:/1.PaidOn/1.FaceRecognition/2.Dataset/2.PaidOnData/Dataset_divi/test/1.src_image',
                        type=str, help='Directory with unaligned images.')
    parser.add_argument('--outdir',
                        default='E:/1.PaidOn/1.FaceRecognition/2.Dataset/2.PaidOnData/Dataset_divi/test/2.image_112x112',
                        type=str, help='Directory with aligned face thumbnails.')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')
    # parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--process', type=int, default=6)
    # parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)

    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png','.bmp'],
                        help='list of acceptable image extensions.')

    parser.add_argument('--recursive', default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    return parser.parse_args(argv)

from multiprocessing import Process
if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    process=[]
    total_folder_number=10175
    boundaries=[total_folder_number//args.process*x for x in range(args.process)]
    boundaries.append(total_folder_number)
    for i in range(args.process):
        process.append(Process(target=main,args=(args,boundaries[i]+1,boundaries[i+1]+1,i)))
    for p in process:
        p.start()
    for p in process:
        p.join()


