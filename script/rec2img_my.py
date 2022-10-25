import mxnet as mx
from mxnet import recordio
import cv2
import os
import argparse
import sys
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--include', default=r"D:\DataBase\51\lfw_masked", type=str, help='')
parser.add_argument('--output', default=r"D:\DataBase\51\lfw_masked\output", type=str, help='')
parser.add_argument('--start',default=1,type=int)
parser.add_argument('--end',default=sys.maxsize,type=int)
args = parser.parse_args()

path_imgidx = os.path.join(args.include,"train.idx") # path to train.rec
path_imgrec = os.path.join(args.include,"train.rec") # path to train.idx
print(args.include)
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#%% 1 ~ 3804847
i=args.start
end=args.end
while True:
        if i >= end: break

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
        path = os.path.join(args.output,str(round(header.label[0])))
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
        cv2.imwrite(path+'.jpg',img)
        if i % 1000 == 0: print(i)
        i += 1