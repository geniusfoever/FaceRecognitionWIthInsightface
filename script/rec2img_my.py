import numpy as np
import mxnet as mx
from mxnet import recordio
import matplotlib.pyplot as plt
import cv2
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default=r"D:\DataBase\51\lfw_masked", type=str, help='')
    parser.add_argument('--output', default=r"D:\DataBase\51\lfw_masked\output", type=str, help='')
    args = parser.parse_args()

path_imgidx = os.path.join(args.include,"train.idx") # path to train.rec
path_imgrec = os.path.join(args.include,"train.rec") # path to train.idx
print(args.include)
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#%% 1 ~ 3804847
i=0
while True:
        print(i)
        try:
            header, s = recordio.unpack(imgrec.read_idx(i))
        except Exception as e:
            print(e)
            print(i)
            break
        #print(str(header.label))
        #img = np.array(mx.image.imdecode(s))
        img = mx.image.imdecode(s).asnumpy()
        #print(type(img))
        path = os.path.join(args.output,str(header.label))
        if not os.path.exists(path):
                os.makedirs(path)
        path = os.path.join(path,str(i))
        #fig = plt.figure(frameon=False)
        #fig.set_size_inches(124,124)
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig.add_axes(ax)
        #ax.imshow(img, aspect='auto')
        #dpi=1
        #fname= str(i)+'jpg'
        #fig.savefig(fname, dpi)
        #plt.savefig(path+'.jpg',bbox_inches='tight',pad_inches=0)
        (b,g,r)=cv2.split(img)
        img = cv2.merge([r,g,b])
        #w,h = img.size
        print((img.shape))
        cv2.imwrite(path+'.jpg',img)
        i+=1
