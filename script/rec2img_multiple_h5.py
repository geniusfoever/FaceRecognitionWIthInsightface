import os
from multiprocessing import Process


import argparse
def run(my_start,my_end,my_include,my_output):
    os.system('python rec2img_my.py --include "{}" --output "{}" --start {} --end {}'.format(my_include,my_output,my_start,my_end))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--process',default=4,type=int)
    parser.add_argument('--start',default=400000,type=int)
    parser.add_argument('--end',default=500000,type=int)
    parser.add_argument('--include', default=r"B:\Database\glint360k", type=str, help='')
    parser.add_argument('--output', default=r"G:\My Drive\insightface\dataset\glink_360m\image", type=str, help='')
    args = parser.parse_args()
    process_list = []
    start = args.start
    one_span = (args.end - start) // args.process
    for i in range(args.process):
        process_list.append(Process(target=run, args=(start, start + one_span, args.include, args.output)))

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

