import IsoGD_Dataset_MSA3D_global as IsoGD_Dataset
from torch.utils.data import DataLoader
import torch
import time
import os
import argparse
import numpy as np
import random
from i3dpt_MSA3D_global import I3D
from i3dpt_MSA3D_global import Unit3Dpy
import torch.optim as optim


def change_lr(opt, learning_rate):
    learning_rate *= 0.1
    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate

def get_init():
    if typ == 'M':
        model = I3D(num_classes=400, modality='rgb')

        if args.rgb_weights_path != '':
            pretrained_dict = torch.load(args.rgb_weights_path)
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict}  # filter out unnecessary keys
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        model.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    print('-----------load model successfully----------')
    return model


def train(dataloader, epoch,log):
    model.train()
    t = time.time()
    correct = 0
    max_num = 0
    for i, (d,dh, n, l) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        outputs,out_logits = model(d.cuda(),dh.cuda())
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == l.cuda()).sum().item()
        max_num += len(d)
        acc = float(correct) / max_num
        loss = criterion1(out_logits, l.cuda())
        loss.backward()
        optimizer.step()
        log.writelines(' Training [%2d/%2d, %4d/%4d] \t Loss: %.4f \t time: %.3f @%s\n' % (epoch, epoch_max_number, i, len(dataloader.dataset)/batch_size, loss.item(), (time.time()-t)/(i+1), time.strftime('%m.%d %H:%M:%S',time.localtime(time.time()))))
        if (i + 1) % 100 == 0:
            print (' Training [%2d/%2d, %4d/%4d] \t Loss: %.4f \t ACC: %.4f \t time: %.3f @%s' % (epoch, epoch_max_number, i, len(dataloader.dataset)/batch_size, loss.item(), acc,(time.time()-t)/(i+1), time.strftime('%m.%d %H:%M:%S',time.localtime(time.time()))))

def test(dataloader, epoch):
    model.eval()
    acc = None
    with torch.no_grad():
        correct = 0
        max_num = 0
        t = time.time()
        for i, (d,dh, n, l) in enumerate(dataloader):

            model.eval()
            outputs ,out_logits= model(d.cuda(),dh.cuda())
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == l.cuda()).sum().item()
            max_num += len(d)
            acc = float(correct) / max_num
            if (i + 1) % 100 == 0:
                print (' Testing [%2d/%2d, %4d/%4d], Acc: %.4f \t time: %.3f' % (epoch, epoch_max_number, i, len(dataloader.dataset)/testing_batch_size, acc, time.time()-t))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='load model')
    parser.add_argument('-m', '--mode', help='train or valid or test', default='train')
    parser.add_argument('-t', '--type', help='M for RGB, K for Depth',default='M')
    parser.add_argument('-g', '--gpu', default="0,1", help="gpu")
    parser.add_argument('--data_root', default="/home/chz/IsoGD/train/")
    parser.add_argument('--hand_data_root', default="/home/chz/IsoGD_hand/train/")
    parser.add_argument('--ground_truth', default="/home/chz/IsoGD/train.txt")
    parser.add_argument('--test_data_root', default="/home/chz/IsoGD/test/")
    parser.add_argument('--test_hand_data_root', default="/home/chz/IsoGD_hand/test/")
    parser.add_argument('--test_ground_truth', default="/home/chz/IsoGD/test.txt")
    parser.add_argument('--save_dir', default="./checkpoints")
    parser.add_argument('--rgb_weights_path', type=str, default='./i3d_pretrained_model/model_rgb.pth',
                        help='Path to rgb model state_dict')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed = 123
    epoch_max_number = 30
    batch_size = 10
    testing_batch_size = 10
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 249
    sn = 16
    mode = args.mode
    typ = args.type
    num_workers = 4

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    log = open('train_MSA3D_global.txt','w')
    if mode != 'train':
        epoch_max_number = 1

    if mode == 'train':
        train_dataset = IsoGD_Dataset.IsoGD_Dataset(args.data_root,
                                                      args.hand_data_root,
                                                      args.ground_truth, typ, sn=sn, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = IsoGD_Dataset.IsoGD_Dataset(args.test_data_root,
                                                     args.test_hand_data_root,
                                                     args.test_ground_truth, typ, sn=sn, phase='test')

        test_dataloader = DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False, num_workers=num_workers)

    model = get_init()
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
    criterion1 = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    save_dir = args.save_dir
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for epoch in range(start_epoch, epoch_max_number):
        if mode == 'train':
            model_name = './{}/MSA3D_global_{}'.format(save_dir, epoch)
        else:
            model_name = args.rgb_weights_path

        if mode == 'train':
            train(train_dataloader, epoch,log)
            test_acc = test(test_dataloader, epoch)
            log.writelines("testAcc:{}\n".format(test_acc))
            torch.save(model, model_name + "_testAcc_{}.model".format(test_acc))
            if (epoch + 1) % 3 == 0:
                learning_rate = change_lr(optimizer, learning_rate)

    log.close()