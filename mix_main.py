import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from dataset.cifar_PLL import load_cifar10,load_cifar100
from dataset.svhn_PLL import load_svhn


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import copy

from utils.parser import set_parser
from utils.function import warmup,eval_train,test,select,test_double,mix_train
from model.wideresnet import WideResNet
from utils.tool import set_seed,get_cosine_schedule

def main():
    args = set_parser()
    set_seed(args)
    
    if args.dataset == 'cifar10' or args.dataset == 'SVHN':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimage':
        args.num_classes = 200
    
    if args.dataset == "cifar10":
        train_all_dataset,test_dataset=load_cifar10(args.partial_rate,root=args.data_path)
    elif args.dataset == "cifar100":
        train_all_dataset,test_dataset=load_cifar100(args.partial_rate,root=args.data_path,hierarchical=args.use_hierarchical)
    elif args.dataset == "SVHN":
        train_all_dataset,test_dataset=load_svhn(args.partial_rate,root=args.data_path+"svhn/")

    if args.arch == 'wideresnet':
        model1 = WideResNet(34, args.num_classes, widen_factor=10, dropRate=0.0)
        model2 = WideResNet(34, args.num_classes, widen_factor=10, dropRate=0.0)
    elif args.arch == 'WRN-28-2':
        model1 = WideResNet(28, args.num_classes, widen_factor=2, dropRate=0.0)
        model2 = WideResNet(28, args.num_classes, widen_factor=2, dropRate=0.0)             
    else:
        assert "Unknown arch"   

    device = torch.device('cuda', args.gpu_id)
    args.device=device 

    model1 = model1.to(args.device)
    model2 = model1.to(args.device)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.StreamHandler()
                    ])

    args.model_name = args.out
    logging.info(args)

    # optimizer
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    # scheduler
    if args.epochs==200:
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[100, 150], last_epoch=-1)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[100, 150], last_epoch=-1)
    
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    train_label_dataset1=copy.deepcopy(train_all_dataset)
    train_label_dataset2=copy.deepcopy(train_all_dataset)
    train_unlabel_dataset=copy.deepcopy(train_all_dataset)

    train_all_loader=DataLoader(
    train_all_dataset,
    sampler=train_sampler(train_all_dataset),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True)

    unlabeled_trainloader = DataLoader(train_unlabel_dataset,
                            sampler = train_sampler(train_unlabel_dataset),
                            batch_size = args.batch_size,
                            num_workers = args.num_workers,
                            drop_last = True)

    n = len(train_all_dataset)
    memory_bank1=np.zeros((args.k,n,args.num_classes))
    memory_bank2=np.zeros((args.k,n,args.num_classes))

    label_matrix_all=copy.deepcopy(train_all_dataset.given_label_matrix)
    label_matrix_all=label_matrix_all.numpy()

    with open(os.path.join("./result/", args.out +
                                  '_training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,selected_ratio1(%),selcted_acc1(%),selected_ratio2(%),selcted_acc2(%),test_loss,test_acc(%)\n')

    selected_ratio1=0
    selcted_acc1=0
    selected_ratio2=0
    selcted_acc2=0
    #main loop
    for epoch in range(args.epochs):
        begin_epoch = time.time()
        if epoch<args.warm_up:
            train_loss1=warmup(args,train_all_loader, model1, optimizer1,epoch)
            train_loss2=warmup(args,train_all_loader, model2, optimizer2,epoch)
            train_loss=(train_loss1+train_loss2)/2
        else:
            clean_idx1,selected_ratio1,selected_label1=select(memory_bank1,label_matrix_all,args)
            clean_idx2,selected_ratio2,selected_label2=select(memory_bank2,label_matrix_all,args)
            # print(selected_ratio)
            args.selected_ratio=(selected_ratio1+selected_ratio2)/2
            train_label_dataset1.init_index()
            train_label_dataset1.set_index(clean_idx2,reture_truelabel=True,selected_labels=selected_label2)

            train_label_dataset2.init_index()
            train_label_dataset2.set_index(clean_idx1,reture_truelabel=True,selected_labels=selected_label1)

            labeled_trainloader1 = DataLoader(train_label_dataset1,
                                        sampler = train_sampler(train_label_dataset1),
                                        batch_size = args.batch_size,
                                        num_workers = args.num_workers,
                                        drop_last = True)
            labeled_trainloader2 = DataLoader(train_label_dataset2,
                                        sampler = train_sampler(train_label_dataset2),
                                        batch_size = args.batch_size,
                                        num_workers = args.num_workers,
                                        drop_last = True)
            train_loss1,selcted_acc1=mix_train(args,labeled_trainloader1, unlabeled_trainloader,model1, optimizer1,epoch)
            train_loss2,selcted_acc2=mix_train(args,labeled_trainloader2, unlabeled_trainloader,model2, optimizer2,epoch)
            train_loss=(train_loss1+train_loss2)/2

        scheduler1.step()
        scheduler2.step()
        memory_bank1 = eval_train(model1, memory_bank1, train_all_loader, args)
        memory_bank2 = eval_train(model2, memory_bank2, train_all_loader, args)
        valloss,valacc  = test_double(args,test_loader,model1,model2)

        with open(os.path.join("./result/", args.out +
                                    '_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            train_loss,
            selected_ratio1,
            selcted_acc1,
            selected_ratio2,
            selcted_acc2,
            valloss,
            valacc       
        ))

if __name__ == '__main__':
    main()