import mxnet as mx
from mxnet import recordio
import cv2
import os
import argparse
import sys
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--include', default=r"B:\Database\glint360k", type=str, help='')
parser.add_argument('--output', default=r"B:\Database\glint360k\image", type=str, help='')
args = parser.parse_args()

path_imgidx = os.path.join(args.include,"train.idx") # path to train.rec
path_imgrec = os.path.join(args.include,"train.rec") # path to train.idx
print(args.include)
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#%% 1 ~ 3804847

def extract(index):

        try:

            header, s = recordio.unpack(imgrec.read_idx(index))


        except Exception as e:
            print(e)
            print(index)
            return

        #print(str(header.label))
        #img = np.array(mx.image.imdecode(s))
        img = mx.image.imdecode(s).asnumpy()
        #print(type(img))
        path = os.path.join(args.output,str(round(header.label[0])))
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
        path = os.path.join(path,str(index))
        if os.path.isfile(path+'.jpg'):
            return
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


from multiprocessing import Pool
from tqdm import tqdm
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    with Pool(processes=4) as pool:
        pool.map(extract, tqdm(range(1, 17091658)),total= 17091657)