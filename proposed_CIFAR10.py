import pickle
import os
import argparse
import logging
import torch
import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils_cifar.loss as al
import utils_cifar.cnn_model as cnn_model
import utils_cifar.calc_hr as calc_hr
import utils_cifar.subset_sampler as subsetsampler

parser = argparse.ArgumentParser(description="proposed")
parser.add_argument('--bits', default='12,24,32,48', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--img_path', default='C:/Users/Ben/Desktop/data', type=str,
                    help='path to image folders (PLS CHANGE)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 0)')
parser.add_argument('--epochs', default=150, type=int,
                    help='number of epochs (default: 150)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size (default: 128)')
parser.add_argument('--alpha', default=0.000001, type=float,
                    help='hyper-parameter: alpha (default: 200)')
parser.add_argument('--beta', default=5 * 10 ** -4, type=float,
                    help='hyper-parameter: beta (default: 5 * 10 ** -4)')
parser.add_argument('--temp1', default=1, type=float,
                    help='hyper-parameter: lambda (default: 1)')
parser.add_argument('--temp2', default=1, type=float,
                    help='hyper-parameter: lambda_0 (default: 1)')
parser.add_argument('--learning-rate', default=0.0001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-4)')
parser.add_argument('--n_class', default=10, type=int,
                    help='number of classes in dataset (default: 10)')
parser.add_argument('--update_list', default='100', type=str,
                    help='epochs where lr is reduced by 90% (default: [100])')


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return

def _dataset(img_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize
    ])
    dset_database = torchvision.datasets.CIFAR10(\
    img_path, train=True,transform =transformations,download=False)

    dset_test = torchvision.datasets.CIFAR10(\
    img_path, train=False,transform =transformations,download=False)

    num_database, num_test = len(dset_database.data), len(dset_test.data)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    return nums, dsets

def encode(model, data_loader, num_data, bit):
    B = np.empty([num_data, bit], dtype=np.float32)
    Y = np.empty([num_data], dtype=np.float32)
    i=0
    for iter, data in enumerate(data_loader, 0):
        data_input, data_labels = data
        s = data_input.shape[0]
        data_input = Variable(data_input.cuda(non_blocking=True))
        output = model(data_input)
        Y[i:(i+s)] = data_labels.numpy()
        B[i:(i+s), :] = torch.sign(output.cpu().data).numpy()
        i+=s
    return B, Y

def adjusting_learning_rate(update_list, optimizer, iter):
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    temp1 = opt.temp1
    temp2 = opt.temp2
    alpha = opt.alpha
    beta = opt.beta
    img_path = opt.img_path
    update_list = [int(epoch) for epoch in opt.update_list.split(',')]

    record['param']['topk'] = 50
    record['param']['topp'] = 100
    record['param']['opt'] = opt
    logger.info(opt)
    logger.info(code_length)

    '''
    dataset preprocessing
    '''
    nums, dsets = _dataset(img_path)
    num_database, num_test = nums
    dset_database, dset_test = dsets

    '''
    model construction
    '''
    model = cnn_model.CNNNet(temp1, code_length, opt.n_class)
    model.cuda()
    proploss = al.proposed(alpha, code_length, temp1, temp2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=beta)
    train_indices = open('data/cifar_train_indices.txt', 'r')
    select_index = [int(x.strip()) for x in train_indices]
    train_indices.close()
    _sampler = torch.utils.data.sampler.SubsetRandomSampler(select_index)
    trainloader = DataLoader(dset_database, batch_size=batch_size,
                             sampler=_sampler,
                             shuffle=False,
                             num_workers=4,pin_memory=True)
    model.train()
    for epoch in range(epochs):
        iter_time = time.time()
        for iteration, (train_input, train_label) in enumerate(trainloader):
            train_input = Variable(train_input.cuda(non_blocking=True))
            train_label= Variable(train_label.float().cuda(non_blocking=True),requires_grad=False)
            output, z, a = model(train_input)
            model.zero_grad()
            loss = proploss(output, z, a, train_label)
            loss.backward()
            optimizer.step()
        adjusting_learning_rate(update_list, optimizer, epoch+1)

        '''
        learning binary codes: discrete coding
        '''
        iter_time = time.time() - iter_time
        loss_ = loss.cpu().data.numpy()
        logger.info('[Iteration: %3d/%3d][Train Loss: %.4f][Time: %.3f secs]', epoch+1, epochs, loss_, iter_time)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

    record['model'] = model
    '''
    training procedure finishes, evaluation
    '''
    model = cnn_model.CNNExtractNet(model)
    model.cuda()
    model.eval()
    trainloader = DataLoader(dset_database, batch_size=128,
                             shuffle=False,
                             num_workers=4,pin_memory=True)
    testloader = DataLoader(dset_test, batch_size=128,
                             shuffle=False,
                             num_workers=4,pin_memory=True)
    qB, test_labels= encode(model, testloader, num_test, code_length)
    rB, database_labels = encode(model, trainloader, num_database, code_length)
    test_indices = open('data/cifar_test_indices.txt', 'r')
    test_index = [int(x.strip()) for x in test_indices]
    test_indices.close()
    gallery_index = np.delete(np.arange(num_test),test_index)
    rB = np.vstack((rB,qB[gallery_index,:]))
    database_labels = np.concatenate((database_labels,test_labels[gallery_index]),axis=0)
    map = calc_hr.calc_map( qB[test_index,:], rB, \
            test_labels[test_index], database_labels)
    logger.info('[Evaluation: MAP: %.4f]', map)
    record['rB'] = rB
    record['qB'] = qB[test_index,:]
    record['MAP'] = map
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)

if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/proposed-cifar10', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        algo(bit)
