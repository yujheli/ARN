from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms as trans
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from reid_network import AdaptReID_model
from reid_dataset import AdaptReID_Dataset
from reid_evaluate import evaluate
from reid_loss import *
import config

parser = ArgumentParser()
parser.add_argument("--use_gpu", default=2, type=int)
parser.add_argument("--source_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--target_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--total_epochs", default=1000, type=int)
parser.add_argument("--w_loss_rec", default=0.1, type=float)
parser.add_argument("--w_loss_dif", default=0.1, type=float)
parser.add_argument("--w_loss_mmd", default=0.1, type=float)
parser.add_argument("--w_loss_ctr", default=1.0, type=float)
parser.add_argument("--dist_metric", choices=['L1', 'L2', 'cosine', 'correlation'], type=str)
parser.add_argument("--rank", default=1, type=int)
parser.add_argument("--model_dir", default='model', type=str)
parser.add_argument("--model_name", default='basic_10cls', type=str)
parser.add_argument("--pretrain_model_name", default=None, type=str)

args = parser.parse_args()
total_batch = 4272
transform_list = [trans.RandomHorizontalFlip(p=0.5), trans.Resize(size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),\
                        trans.ToTensor(), trans.Normalize(mean=config.MEAN, std=config.STD)]

def setup_gpu():
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Using GPU: {}'.format(args.use_gpu))

def tb_visualize(image):
    return (image+1.0)*0.5

def get_batch(data_iter, data_loader):
    try:
        _, batch = next(data_iter)
    except:
        data_iter = enumerate(data_loader)
        _, batch = next(data_iter)
    if batch['image'].size(0) < args.batch_size:
        batch, data_iter = get_batch(data_iter, data_loader)
    return batch, data_iter

def save_model(model, step):
    model_path = '{}/{}/{}_{}.pth.tar'.format(args.model_dir ,args.model_name ,args.model_name, step)
    torch.save(model.state_dict(), model_path)

def train():
    print('Model name: {}'.format(args.model_name))    
    classifier_output_dim = config.get_dataoutput(args.source_dataset)
    model = AdaptReID_model(backbone='resnet-50', classifier_output_dim=classifier_output_dim).cuda()
    # print(model.state_dict())

    print("Loading pre-trained model...")
    checkpoint = torch.load('{}/{}.pth.tar'.format(args.model_dir, args.pretrain_model_name))
    for name, param in model.state_dict().items():
        print(name)
        if name not in ['classifier.linear.weight','classifier.linear.bias']:
            model.state_dict()[name].copy_(checkpoint[name]) 

    # if args.pretrain_model_name is not None:
    #     model.load_state_dict(torch.load('{}/{}.pth.tar'.format(args.model_dir, args.pretrain_model_name)))

    sourceData = AdaptReID_Dataset(dataset_name=args.source_dataset, mode='source', transform=trans.Compose(transform_list))
    targetData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='train', transform=trans.Compose(transform_list))
    sourceDataloader = DataLoader(sourceData, batch_size=args.batch_size, shuffle=True)
    targetDataloader = DataLoader(targetData, batch_size=args.batch_size, shuffle=True)
    source_iter = enumerate(sourceDataloader)
    target_iter = enumerate(targetDataloader)

    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    writer = SummaryWriter('log/{}'.format(args.model_name))
    match, junk = None, None

    for epoch in range(args.total_epochs):
        for batch_idx in range(total_batch):
            model.train()
            step = epoch*total_batch + batch_idx
            print('{}: epoch: {}/{} step: {}/{} ({:.2f}%)'.format(args.model_name, epoch+1, args.total_epochs, step+1, total_batch, float(batch_idx+1)*100.0/total_batch))
            source_batch, source_iter = get_batch(source_iter, sourceDataloader)
            target_batch, target_iter = get_batch(target_iter, targetDataloader)
            source_image, source_label, _ = split_datapack(source_batch)
            target_image, _, _ = split_datapack(target_batch)
            
            feature_t_et_avg, feature_t_ec_avg, image_t_, pred_t, feature_s_es_avg, feature_s_ec_avg, image_s_, pred_s\
                         = model(image_t=target_image, image_s=source_image)
            
            #calculate loss
            loss_rec = loss_rec_func(image_s_, source_image) + loss_rec_func(image_t_, target_image)
            loss_cls = loss_cls_func(pred_s, source_label)
            loss_dif = loss_dif_func(feature_t_et_avg, feature_t_ec_avg) + loss_dif_func(feature_s_es_avg, feature_s_ec_avg)
            loss_mmd = loss_mmd_func(feature_t_ec_avg, feature_s_ec_avg)
            loss_ctr = loss_ctr_func(feature_s_ec_avg, source_label)
            loss = loss_cls + loss_rec*args.w_loss_rec + loss_dif*args.w_loss_dif + loss_ctr*args.w_loss_ctr + loss_mmd*args.w_loss_mmd

            #update model
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()

            #write to tenserboardX
            if (step+1)%100==0:
                writer.add_image('source image', tb_visualize(source_image), step)
                writer.add_image('target image', tb_visualize(target_image), step)
                writer.add_image('source image_rec', tb_visualize(image_s_), step)
                writer.add_image('target image_rec', tb_visualize(image_t_), step)
                writer.add_scalar('loss_rec', loss_rec, step)
                writer.add_scalar('loss_cls', loss_cls, step)
                writer.add_scalar('loss_dif', loss_dif, step)
                writer.add_scalar('loss_ctr', loss_ctr, step)
                writer.add_scalar('loss_mmd', loss_ctr, step)
                writer.add_scalar('loss', loss, step)

            #evaluate
            if (step+1)%2000==0:
                rank_score, match, junk = evaluate(args, model, transform_list, match, junk)
                writer.add_scalar('rank1_score', rank_score, step)
                save_model(model, step)

    writer.close()
   
setup_gpu()
train() 
