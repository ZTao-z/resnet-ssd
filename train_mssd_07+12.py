from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, VOC_320, VOC_300_1, VOC_300_2, COCO_300, COCO_512, \
    COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import time

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='ssd300',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version or ssd.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=2,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=250,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
elif args.version == 'ssd_ori':
    from models.ssd_ori import build_net

    cfg = VOC_300_1
elif args.version == 'ssd':
    from models.ssd import build_net
    cfg = VOC_320
elif args.version == 'ssd300':
    from models.res_34 import build_net
    cfg = VOC_300_2
else:
    print('Unkown version!')

img_dim = (300, 512)[args.size == '512']
if (args.size == '320'):
    img_dim = 320
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes)
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    #net.vgg.load_state_dict(base_weights)
    model_dict = net.state_dict()
    resnet18_para = torch.load("weights/resnet_34_para.pth")
    # ssd_net.load_state_dict(resnet18_para)

    pretrained_dict = resnet18_para
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    net.load_state_dict(model_dict)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0
    def weights_init1(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.vgg1.apply(weights_init)
    net.vgg2.apply(weights_init)
    #net.vgg3.apply(weights_init)
    #net.vgg4.apply(weights_init)
    net.vgg5.apply(weights_init)
    net.vgg6.apply(weights_init)
    net.vgg7.apply(weights_init)
    net.vgg8.apply(weights_init)
    net.de1.apply(weights_init)
    net.de2.apply(weights_init)
    net.de3.apply(weights_init)
    net.de4.apply(weights_init)
    net.ds5_3.apply(weights_init)
    net.ds10_5.apply(weights_init)
    net.ds19_10.apply(weights_init)
    net.ds38_19.apply(weights_init)
    #net.de5_19.apply(weights_init)
    #net.de10_38.apply(weights_init)
    net.extras.apply(weights_init)
    net.conf.apply(weights_init)
    net.loc.apply(weights_init)
    net.d19sample1.apply(weights_init1)
    net.d19sample2.apply(weights_init1)
    net.d19sample3.apply(weights_init1)
    net.con_press38.apply(weights_init)
    #net.Norm.apply(weights_init)
    #if args.version == 'RFB_E_vgg':
        #net.reduce.apply(weights_init)
        #net.up_reduce.apply(weights_init)

else:
# load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
if args.cuda:
    net = torch.nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150*epoch_size, 200*epoch_size, 250*epoch_size)  # (80000,100000,120000)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            #if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                #torch.save(net.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
                           #repr(epoch) + '.pth')
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        load_t0 = time.time()
        # t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        # t1 = time.time()
        load_t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 100 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      loss_l.data[0], loss_c.data[0]) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0)+ 'LR: %.8f' % (lr))
        if iteration<=110000 and (iteration == 0 or iteration % 10000 == 0):
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/ssd300_2_VOC_' +
                       repr(iteration) + '.pth')
        elif (iteration>110000) and  iteration % 1000 == 0 :
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/ssd300_2_VOC_' +
                       repr(iteration) + '.pth')



    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
if __name__ == '__main__':
    train()
