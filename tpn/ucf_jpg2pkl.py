import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


label_dic = np.load('ucf101_label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = '/home/aistudio/UCF-101-jpg/'
target_train_dir = source_dir + 'train'
target_test_dir = source_dir + 'test'
target_val_dir = source_dir + 'val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

for key in label_dic:
    each_mulu = key
    print(each_mulu, key) 

    label_dir = os.path.join(source_dir, each_mulu) #一级类别 hair_cut 
    label_mulu = os.listdir(label_dir)
    tag = 1

    cur_len = len(label_mulu)
    split_len = int(cur_len*0.1)
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu)) #二级 work/UCF-101-jpg/Haircut/v_Haircut_g16_c01
        image_file.sort()

        image_num = len(image_file)
        frame = []
        vid = each_label_mulu
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_file[i])
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if tag < cur_len - 2*split_len:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif tag < cur_len - split_len:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        else:
            output_pkl = os.path.join(target_val_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
