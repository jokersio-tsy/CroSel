import argparse
from cProfile import label
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import copy

from utils.tool import AverageMeter, accuracy
from utils.loss import *

logger = logging.getLogger(__name__)

def select(memory_bank,label_matrix,args):
    #use hard label
    max_prob=np.max(memory_bank,axis=2)
    # print(max_prob)
    pred=np.argmax(memory_bank,axis=2)
    # max_prob,pred=torch.max(memory_bank,dim=2)
    pred_last=pred[-1]
    n=pred.shape[1]
    bool_equal=np.zeros((args.k-1,n))
    for i in range(args.k-1):
        bool_equal[i]=np.equal(pred[i],pred[i+1])
    sum_equal=np.sum(bool_equal,axis=0)
    jud1= sum_equal == args.k-1
    max_prob_mean=np.mean(max_prob,axis=0)
    jud2= max_prob_mean > args.select_threshold
    jud3=np.zeros(n)
    for i in range(n):
        jud3[i]= label_matrix[i][pred_last[i]]
    jud_tmp=np.concatenate([jud1,jud2,jud3],axis=-1)
    jud_tmp=jud_tmp.reshape(3,n)
    jud=np.sum(jud_tmp,axis=0)
    clean_ind = np.where(jud == 3)[0]

    return clean_ind ,len(clean_ind)/n, pred_last[clean_ind]

def test_double(args, test_loader, model1,model2):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model1.eval()
            model2.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs=(outputs1+outputs2)/2
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    batch_idx, len(test_loader), batch_time=batch_time, loss=losses,top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    batch_idx, len(test_loader), batch_time=batch_time, loss=losses,top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def eval_train(model, memory_bank, eval_loader, args):
    model.eval()
    n = len(eval_loader.dataset)

    result = torch.zeros(n,args.num_classes)
    # result_predict =torch.zeros(n)
    with torch.no_grad():
        for i, ((images_w,images_s,images_w2),labels, index) in enumerate(eval_loader):
            inputs, targets = images_w2.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            sm_outputs= F.softmax(outputs,1)
            max_probs, pred = outputs.max(1)

            for b in range(inputs.size(0)):
                sample_index = index[b]
                result[sample_index]=sm_outputs[b]

    memory_bank_tmp=memory_bank.copy()
    memory_bank[:args.k-1]=memory_bank_tmp[1:]
    memory_bank[-1] = result.numpy()
    del memory_bank_tmp

    return memory_bank

def warmup(args,train_loader, model, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for i, ((images_w,images_s,images_w2),label_matrix, index) in enumerate(train_loader):
        inputs, label_matrix = images_w2.to(args.device), label_matrix.to(args.device)
        outputs = model(inputs)
        loss=cc_loss(outputs,label_matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,loss=losses))
    
    return losses.avg

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def mix_train(args,labeled_trainloader, unlabeled_trainloader,model, optimizer,epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    pesudo_cor=0
    pesudo_sum=0
    # for batch_idx in range(args.eval_step):
    for batch_idx, ((input_u_w,input_u_s,inputs_u_w2),label_matrix_all,idx) in enumerate(unlabeled_trainloader):
        try:
            (input_l_w,input_l_s,input_l_w2),label_matrix,true_label,selected_label,idx = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_trainloader)
            (input_l_w,input_l_s,input_l_w2),label_matrix,true_label,selected_label,idx = next(labeled_iter)

        batch_size = input_l_w.shape[0]

        input_l_w=input_l_w.to(args.device)
        logits_x = model(input_l_w)
        inputs = torch.cat((input_u_w, input_u_s),dim=0).to(args.device)
        inputs_target= torch.cat((input_u_s, input_u_w),dim=0).to(args.device)
        label_matrix_all=torch.cat((label_matrix_all,label_matrix_all),0).to(args.device)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_target)
            outputs_u=torch.softmax(outputs_u,dim=1)
            pt = outputs_u**(1/args.sharpen_T)
            pt = outputs_u* label_matrix_all
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # test seleted_acc
        label_matrix = label_matrix.to(args.device)
        true_label = true_label.to(args.device)
        selected_label = selected_label.to(args.device)
        # label_matrix_all = label_matrix_all.to(args.device)
        pesudo_cor+=(selected_label == true_label).sum().item()
        pesudo_sum+=true_label.shape[0]

        batch_time.update(time.time() - end)
        end = time.time()

        #label loss
        Lx = F.cross_entropy(logits_x, selected_label.long(), reduction='mean')

        if args.use_mix == True:            
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
            idx = torch.randperm(inputs.size(0))
            input_a, input_b = inputs, inputs[idx]
            target_a, target_b = targets_u, targets_u[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b
        else:
            mixed_input=inputs.clone()
            mixed_target=targets_u.clone()

        logits=model(mixed_input)
        logits_u_log_softmax=F.log_softmax(logits, dim=1)
        Lcr = -torch.mean(torch.sum(logits_u_log_softmax * mixed_target, dim=1))
            
        dynamic_lambda=1.0-args.selected_ratio
        # dynamic_lambda=1.0
        loss=Lx +  dynamic_lambda * args.lambda_cr * Lcr
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, batch_idx, args.eval_step, batch_time=batch_time,loss=losses))

    return losses.avg, pesudo_cor/(pesudo_sum+1)*100