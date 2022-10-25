import os
from multiprocessing import Process


import argparse
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--process',default=32,type=int)
parser.add_argument('--start',default=1,type=int)
parser.add_argument('--end',type=int)
parser.add_argument('--include', default=r"D:\DataBase\51\lfw_masked", type=str, help='')
parser.add_argument('--output', default=r"D:\DataBase\51\lfw_masked\output", type=str, help='')
args = parser.parse_args()

def run(my_start,my_end,my_include=args.include,my_output=args.output):
    os.system("rec2img_my.py --include {} --output {} --start {} --end {}".format(my_include,my_output,my_start,my_end))

process_list=[]
start=args.start
one_span=(args.end-start)//args.process
for i in range(args.process):
    process_list.append(Process(target=run, args=(start,start+one_span,args.include,args.output)))
for p in process_list:
    p.start()
for p in process_list:
    p.join()