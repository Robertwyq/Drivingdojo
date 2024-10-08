import os
import numpy as np
from PIL import Image
import argparse
import multiprocessing
import tensorflow.compat.v1 as tf

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
tf.enable_eager_execution()

def read_root(root): 
    file_list = sorted(os.listdir(root))
    return file_list

def main(waymo_root,split,output_dir,token, process_num, debug):
    train_root = os.path.join(waymo_root,split)

    file_list = read_root(train_root)
    if debug:
        file_list = file_list[0:5]

    for s in range(len(file_list)):
        if s % process_num != token:
            continue
        filename = file_list[s]
        FILENAME = os.path.join(train_root,filename)
        segment_dir = os.path.join(output_dir,filename.split('.')[0])
        os.makedirs(output_dir,exist_ok=True)
        segment_img_dir = os.path.join(segment_dir,'CAM_FRONT')
        os.makedirs(segment_img_dir,exist_ok=True)

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_list = ["%06d" % (x) for x in range(200)]
        # jpg,png (img format)
        new_imgs = ['frame_'+t+'.jpg' for t in frame_list]
        for idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # only extract front-view img 
            for index, image in enumerate(frame.images):
                img = tf.image.decode_jpeg(image.image)
                img = np.array(img)
                I = Image.fromarray(np.uint8(img))
                save_path = os.path.join(segment_img_dir,new_imgs[idx])
                I.save(save_path)
                break
            
            
        print('finish:',filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/mnt/nvme1n1p1/waymo',help='path to the waymo open dataset')
    parser.add_argument('--split',default='training',choices=['training','validation'])
    parser.add_argument('--output_dir',default='/mnt/nvme0n1p1/yuqi.wang/datasets/waymo_video',help='path to save the data')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    parser.add_argument('--debug', type=bool, default=False, help = 'only test for 5 segments')
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root,args.split,args.output_dir, token, args.process,args.debug))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root,args.split,args.output_dir, 0, args.process, args.debug)
