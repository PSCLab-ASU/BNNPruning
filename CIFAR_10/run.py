from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import time
import torch.nn as nn
import torch.optim as optim
import pickle

from copy import deepcopy

from models import nin
from torch.autograd import Variable

torch.cuda.set_device(3)

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    #act_cnt = 0
    act_cnt = torch.zeros(100,144,16,16).cuda()
    #act_cnt = torch.sparse.FloatTensor(100,144,16,16)
    cnt = 0

    bin_op.binarization()
    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        #print(output)
        activations0 = model.xnor1.forward(data)
        activations1 = model.xnor2.forward(activations0)
        activations2 = model.xnor3.forward(activations1)
        print(activations2.size())
        #print(act_cnt.size())
        #torch.sparse.FloatTensor(activations2).to_dense()
        # if cnt == 0:
        #     act_cnt = ((activations2-0.5).sign() + 1)/2
        #     act_cnt.cuda()
        # else:
        #     act_cnt = act_cnt + ((activations2-0.5).sign() + 1) / 2
        tp = ((activations2 - 0.5).sign() + 1) / 2
        tp1 = tp.data
        print(type(tp1))
        print(type(act_cnt))
        act_cnt = act_cnt + tp1
        cnt = cnt + 1
        #print(activations2)
        print(cnt)
        print(act_cnt)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * correct / len(test_loader.dataset)

####
#activations = model.xnor1.forward(test_loader.dataset.test_data[:, :, :, :])
####


    if acc > best_acc:
        best_acc = acc

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def test2(evaluate=False):
    global best_acc

    model.eval()
    test_loss = 0
    correct = 0

    bin_op.binarizationTest()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    bin_op.restore()

    acc = 100. * correct / len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return



if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default='models/nin.pth.tar',
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='whether to run evaluation')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    trainset = data.dataset(root=args.data, train=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    testset = data.dataset(root=args.data, train=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = nin.Net()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        # use all GPUs
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # use 2 GPUs
        #model = torch.nn.DataParallel(model, device_ids=[1])
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    flip_mat = [0] * 7
    flip_mat_sum = [0] * 7
    target_modules_last = [0] * 7
    flip_mat_mask = [0] * 7
    flip_mat_mask_tp = [0] * 7
    random_mat_mask = [0] * 7

    # define the binarization operator
    bin_op = util.BinOp(model, flip_mat_mask)

    # do the evaluation if specified
    if args.evaluate:
        test()
        #exit(0)

    with open("flip_mat_sum.txt", "rb") as fp:
        flip_mat_sum = pickle.load(fp)

    with open("flip_mat_sum_epoch300.txt", "rb") as fp2:
        flip_mat_sum_300 = pickle.load(fp2)

    for index in range(bin_op.num_of_params):
        flip_mat_sum[index] = flip_mat_sum[index] - flip_mat_sum_300[index]



    print('final test')
    for index in range(bin_op.num_of_params):
        tp = flip_mat_sum[index] - 0.5
        flip_mat_mask_tp[index] = (-1 * tp.sign() + 1) / 2
        #bin_op.target_modules[index].data = bin_op.target_modules[index].data * flip_mat_mask[index]
        tp2 = torch.rand(flip_mat_mask_tp[index].size()).cuda()
        random_mat_mask[index] = ((tp2 - 0.5).sign() + 1) / 2
        random_mat_mask[index] = random_mat_mask[index] * (torch.ones(flip_mat_mask_tp[index].size()).cuda() - flip_mat_mask_tp[index])
        flip_mat_mask[index] = flip_mat_mask_tp[index] + random_mat_mask[index]
    test2()


