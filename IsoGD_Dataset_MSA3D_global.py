#-*- coding: utf-8 -*-
"""
HA_RPSSDataset, Dataset for SDSATT Training
"""
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
import random
import cv2
import os, sys
import numpy as np
#from scipy.ndimage.filters import gaussian_filter

class IsoGD_Dataset(Dataset):
    def __init__(self, dataset_root,hand_root, ground_truth, typ, sn=16, phase = 'train'):
        def get_data_list_and_label(dataset_root,data_df, typ):
            self.T = 0  if typ == 'M' else 1
            return [(lambda arr: (dataset_root+'/'.join(arr[0].split('/')[1:]),dataset_root+ '/'.join(arr[1].split('/')[1:]), int(arr[2])))(
                i[:-1].split(' '))
                    for i in open(data_df).readlines()]


        if phase=='train':
            # self.data_list = get_data_list_and_label('/home/s01/syy/Dataset/IsoGD/valid/', '/home/s01/syy/Dataset/IsoGD/valid.txt', typ)

            # self.data_list=get_data_list_and_label(dataset_root, ground_truth, typ)+get_data_list_and_label('/home/chenhuizhou/IsoGD/valid/','/home/chenhuizhou/IsoGD/valid.txt', typ)
            self.data_list = get_data_list_and_label(dataset_root, ground_truth, typ)
        else:
            self.data_list = get_data_list_and_label(dataset_root, ground_truth, typ)
        # print(get_data_list_and_label('/home/s01/syy/Dataset/IsoGD/valid/','/home/s01/syy/Dataset/IsoGD/valid.txt', typ))
        # print(len(self.data_list))
        # print(self.data_list)
        self.dataset_root = dataset_root
        self.hand_root = hand_root
        self.phase = phase
        self.sn = sn
        self.typ=typ
    def transform_params(self, resize = (320, 240), crop_size = 224, flip = 0.5):
        if self.phase == 'train':
            left, top = random.randint(0, resize[0] - crop_size), random.randint(0, resize[1] - crop_size)
            is_flip = True if random.uniform(0, 1) > flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def __getitem__(self, index):

        resize = (320, 240) # default | (256, 256) may be helpful
        #resize = (256, 256)
        crop_rect, is_flip = self.transform_params(resize = resize, flip = 1.0) # no flip
        #print (crop_rect, is_flip)
        def transform(img):
            def flip(data, is_flip):
                if is_flip:
                    data = data.transpose(Image.FLIP_LEFT_RIGHT)
                return data

            img = np.array(flip(Image.fromarray(img).resize(resize).crop(crop_rect), is_flip))

            #img = np.array(img)
            img -= np.min(img)
            img = 255.0 * img / (np.max(img)+10e-7)


            '''
            #visualize
            import matplotlib.pyplot as plt
            plt.figure('test')
            plt.imshow(img.astype('uint8').squeeze())
            plt.show()
            '''

            return img.astype('uint8')

        def sampling(arr, sn = self.sn):

            if self.phase == 'train':
                f = lambda n: [(lambda n, arr: n if arr==[] else
                    random.choice(arr))(n*i/sn, range(int(n*i/sn), max(int(n*i/sn)+1, int(n*(i+1)/sn)))) for i in range(sn)]
            else:
                f = lambda n: [(lambda n, arr: n if arr==[] else
                    int(np.mean(arr)))(n*i/sn, range(int(n*i/sn), max(int(n*i/sn)+1, int(n*(i+1)/sn)))) for i in range(sn)]

            sl = f(len(arr))
            return [transform(arr[i]) for i in sl], sl

        arr = []
        hand_arr = []
        #print(self.data_list[index][self.T])
        # data_path = self.dataset_root + '/' + self.data_list[index][self.T]
        data_path = self.data_list[index][self.T]
        # print(data_path)
        hand_path = self.hand_root[:-6]+'/'+ '/'.join(data_path.split('/')[-3:])
        # print(data_path,hand_path)
        cap = cv2.VideoCapture(data_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                #if self.typ =='K':
                #    arr.append(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
                    
                #else:
                arr.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # the opencv read image is in BGR order
            else:
                break
        cap.release()
        hand = cv2.VideoCapture(hand_path)
        while (hand.isOpened()):
            ret, frame = hand.read()
            if ret == True:
                #if self.typ =='K':
                #    arr.append(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
                    
                #else:
                hand_arr.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # the opencv read image is in BGR order
            else:
                break
        hand.release()
        # olderr = np.seterr(all='ignore')

        arr, sl = sampling(arr)
        hand_arr=[transform(hand_arr[i]) for i in sl]
        #if self.typ=='M':
        #    arr = [transforms.ToTensor()(frame).view(3, 224, 224, 1) for frame in arr]
        #else:
        #    arr = [transforms.ToTensor()(frame).view(1, 224, 224, 1) for frame in arr]
        arr = [transforms.ToTensor()(frame).view(3, 224, 224, 1) for frame in arr]
        hand_arr = [transforms.ToTensor()(frame).view(3, 224, 224, 1) for frame in hand_arr]
        # arr = [transforms.ToTensor()(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)).view(3, 224, 224, 1) for frame in arr]
        arr = torch.cat(arr, dim = 3)
        hand_arr = torch.cat(hand_arr, dim = 3)
        """
        def tensor_split(t):
            a = torch.split(t, 1, dim=3) # dim = 4 is time dimension
            a = [x.view(x.size()[:-1]) for x in a]
            return a
        for i, x in enumerate(tensor_split(arr)):
            img = transforms.ToPILImage()(x)            
            img.save("{}_{}.jpg".format(index, i))
        """
        #print(self.data_list[0])
        return arr,hand_arr, self.data_list[index][0], self.data_list[index][2]-1

    def __len__(self):
        return len(self.data_list)
