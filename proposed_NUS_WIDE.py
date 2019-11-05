import pickle
import os
import argparse
import logging
import torch
import time
import pdb

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.loss as al
import utils.cnn_model as cnn_model
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="proposed")
parser.add_argument('--bits', default='8,16,24,32', type=str,
                    help='binary code length (default: 8,16,24,32)')
parser.add_argument('--img_path', default='C:/Users/Ben/Desktop/NUS_WIDE/image', type=str,
                    help='path to image folders (PLS CHANGE)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 0)')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of epochs (default: 20)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--alpha', default=10 ** -6, type=float,
                    help='hyper-parameter: alpha (default: 10 ** -6)')
parser.add_argument('--beta', default=5 * 10 ** -4, type=float,
                    help='hyper-parameter: beta (default: 5 * 10 ** -4)')
parser.add_argument('--temp1', default=1, type=float,
                    help='hyper-parameter: lambda (default: 1)')
parser.add_argument('--temp2', default=1, type=float,
                    help='hyper-parameter: lambda_0 (default: 1)')
parser.add_argument('--learning-rate', default=0.0001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-4)')
parser.add_argument('--n_class', default=21, type=int,
                    help='number of classes in dataset (default: 21)')

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

    dset_database = dp.DatasetProcessingNUS_WIDE(
        img_path,'data/NUS-WIDE', 'database_img.txt', 'database_label.txt', transformations
    )
    dset_test =  dp.DatasetProcessingNUS_WIDE(
        img_path,'data/NUS-WIDE', 'test_img.txt', 'test_label.txt', transformations
    )
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        label_filepath = os.path.join(DATA_DIR, filename)
        label = np.loadtxt(label_filepath, dtype=np.int64)
        return torch.from_numpy(label)

    databaselabels = load_label('database_label.txt', 'data/NUS-WIDE')
    testlabels = load_label('test_label.txt', 'data/NUS-WIDE')

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda(non_blocking=True))
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

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

    record['param']['topk'] = 50
    record['param']['topp'] = 100
    record['param']['opt'] = opt
    logger.info(opt)
    logger.info(code_length)

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset(img_path)
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(temp1, code_length, opt.n_class)
    model.cuda()
    proploss = al.proposed(alpha, code_length, temp1, temp2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=beta)
    trainloader = DataLoader(dset_database, batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,pin_memory=True)
    model.train()
    for epoch in range(epochs):
        iter_time = time.time()
        for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
            train_input = Variable(train_input.cuda(non_blocking=True))
            train_label= Variable(train_label.float().cuda(non_blocking=True),requires_grad=False)
            output, z, m = model(train_input)
            model.zero_grad()
            loss = proploss(output, z, m, train_label)
            loss.backward()
            optimizer.step()

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
    model.eval()
    trainloader = DataLoader(dset_database, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,pin_memory=True)
    testloader = DataLoader(dset_test, batch_size=1,
                             shuffle=False,
                             num_workers=4,pin_memory=True)
    qB = encode(model, testloader, num_test, code_length)
    rB = encode(model, trainloader, num_database, code_length)
    mapka, mapkb, topp_ndcg, topp_acg = calc_hr.calc_metrics(qB, rB, test_labels.numpy(), database_labels.numpy(),
                                                                record['param']['topk'],
                                                                record['param']['topp'],
                                                                record['param']['topp'])
    logger.info('[Evaluation: mAP@%d (A): %.4f, mAP@%d (B): %.4f, top-%d NDCG: %.4f, top-%d ACG: %.4f]',
                                                     record['param']['topk'], mapka,
                                                     record['param']['topk'], mapkb,
                                                     record['param']['topp'], topp_ndcg,
                                                     record['param']['topp'], topp_acg)
    record['rB'] = rB
    record['qB'] = qB
    record['mapka'] = mapka
    record['mapkb'] = mapkb
    record['topp_ndcg'] = topp_ndcg
    record['topp_acg'] = topp_acg
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)

if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/proposed-nuswide', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        algo(bit)
