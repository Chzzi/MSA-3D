#-*- coding: utf-8 -*-
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
import cv2
import numpy as np

class IsoGD_Dataset(Dataset):
    def __init__(self, dataset_root, ground_truth, typ, sn=16, phase = 'train'):
        def get_data_list_and_label(data_df, typ):
            self.T = 0  if typ == 'rgb' else 1
            return [(lambda arr: ('/'.join(arr[0].split('/')[1:]), '/'.join(arr[1].split('/')[1:]), int(arr[2])))(
                i[:-1].split(' '))
                    for i in open(data_df).readlines()]
        if typ == 'M':
            self.typ = 'rgb'
        else:
            self.typ = 'depth'
        self.data_list = get_data_list_and_label(ground_truth, typ)
        self.dataset_root = dataset_root
        self.phase = phase
        self.sn = sn

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
        crop_rect, is_flip = self.transform_params(resize = resize, flip = 1.0) # no flip
        def transform(img):
            def flip(data, is_flip):
                if is_flip:
                    data = data.transpose(Image.FLIP_LEFT_RIGHT)
                return data

            img = np.array(flip(Image.fromarray(img).resize(resize).crop(crop_rect), is_flip))

            img -= np.min(img)
            img = 255.0 * img / (np.max(img) + 0.000001)

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
        data_path = self.dataset_root + '/' + self.data_list[index][self.T]
        cap = cv2.VideoCapture(data_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if self.typ == 'rgb':
                    arr.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    arr.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break
        cap.release()
        arr, sl = sampling(arr)
        if self.typ=='rgb':
           arr = [transforms.ToTensor()(frame).view(3, 224, 224, 1) for frame in arr]
        else:
           arr = [transforms.ToTensor()(np.expand_dims(frame,2).repeat(3, axis=2)).view(3, 224, 224, 1) for frame in arr]
        arr = torch.cat(arr, dim = 3)
        return {'data': arr, 'label': self.data_list[index][2] - 1}

    def __len__(self):
        return len(self.data_list)
