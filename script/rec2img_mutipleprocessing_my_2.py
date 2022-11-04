import mxnet as mx
from mxnet import recordio
import cv2
import os
import argparse
import sys
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--include', default=r"B:\Database\glint360k", type=str, help='')
parser.add_argument('--output', default=r"D:\DataBase\Glint360k", type=str, help='')
args = parser.parse_args()
output=args.output
path_imgidx = os.path.join(args.include,"train.idx") # path to train.rec
path_imgrec = os.path.join(args.include,"train.rec") # path to train.idx

def extract(start,end,p_id):
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    header, s = recordio.unpack(imgrec.read_idx(start))
    img = mx.image.imdecode(s).asnumpy()
    # print(type(img))
    path = os.path.join(output, str(round(header.label[0])))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    path = os.path.join(path, str(header.id))

    (b, g, r) = cv2.split(img)
    img = cv2.merge([r, g, b])
    # w,h = img.size
    cv2.imwrite(path + '.jpg', img)

    for i in tqdm(range(start,end),position=p_id):
        header, s = recordio.unpack(imgrec.read())

        #print(str(header.label))
        #img = np.array(mx.image.imdecode(s))
        img = mx.image.imdecode(s).asnumpy()
        #print(type(img))
        path = os.path.join(output,str(round(header.label[0])))
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
        path = os.path.join(path,str(header.id))
        if os.path.isfile(path+'.jpg'):
            continue
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


from multiprocessing import Process
from tqdm import tqdm
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    process=8
    total_number=17091657
    start_end_list=[total_number//process*i for i in range(process)]
    start_end_list.append(total_number)
    start_end_list[0]=1
    process_list=[]
    for i in range(process):
        process_list.append(Process(target=extract,args=(start_end_list[i],start_end_list[i+1],i)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
