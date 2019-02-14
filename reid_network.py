from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class Base_Encoder(nn.Module):
    def __init__(self, backbone='resnet-50'):
        super(Base_Encoder, self).__init__()
        if backbone == 'resnet-50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-3])

    def forward(self, input_img):
        return self.model(input_img)

class Encoder(nn.Module):
    def __init__(self, backbone='resnet-50'):
        super(Encoder, self).__init__()
        if backbone == 'resnet-50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[7:-2])
        self.avgpool = nn.AvgPool2d((8,4))

    def forward(self, input_feature, use_avg=False):
        feature = self.model(input_feature)
        if use_avg:
            feature = self.avgpool(feature)
            feature = feature.view(feature.size()[0],-1)
        return feature

class Decoder(nn.Module):
    def __init__(self, ch_list=None):
        super(Decoder, self).__init__()

        def get_layers(in_filters, out_filters, stride=1, out_pad=0):
            layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3,\
                        stride=stride, padding=1, output_padding=out_pad),
                     nn.BatchNorm2d(out_filters),
                     nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return layers
        self.block1 = nn.Sequential(
            *get_layers(ch_list[0], ch_list[1], stride=2, out_pad=1),
            *get_layers(ch_list[1], ch_list[1]),
            *get_layers(ch_list[1], ch_list[1])
        )

    def forward(self, input_feature):
        return self.block1(input_feature)

class Base_Decoder(nn.Module):
    def __init__(self, ch_list=None):
        super(Base_Decoder, self).__init__()
        def get_layers(in_filters, out_filters, stride=1, out_pad=0, last=False):
            layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3,\
                        stride=stride, padding=1, output_padding=out_pad),
                     nn.BatchNorm2d(out_filters)]
            if last:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        def make_blocks(in_ch, out_ch, last=False):
            block = nn.Sequential(
                *get_layers(in_ch, out_ch, stride=2, out_pad=1),
                *get_layers(out_ch, out_ch),
                *get_layers(out_ch, out_ch, last=last)
            )
            return block

        self.block1 = make_blocks(ch_list[0], ch_list[1])
        self.block2 = make_blocks(ch_list[1], ch_list[2])
        self.block3 = make_blocks(ch_list[2], ch_list[3])
        self.block4 = make_blocks(ch_list[3], ch_list[4])
        self.block5 = make_blocks(ch_list[4], ch_list[5], last=True)

    def forward(self, input_feature):
        feature1 = self.block1(input_feature)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        feature4 = self.block4(feature3)
        feature5 = self.block5(feature4)
        return feature5

class Classifier(nn.Module):
    def __init__(self, input_dim=2048, output_dim=-1):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, feature):
        feature = self.dropout(feature)
        prediction = self.linear(feature)
        return prediction

class AdaptReID_model(nn.Module):
    def __init__(self, backbone='resent-50', \
        classifier_input_dim=2048, classifier_output_dim=-1):
        super(AdaptReID_model, self).__init__()
        
        self.encoder_base = Base_Encoder(backbone=backbone)
        self.encoder_t = Encoder(backbone=backbone)
        self.encoder_c = Encoder(backbone=backbone)
        self.encoder_s = Encoder(backbone=backbone)

        self.ch_list = [2048, 1024, 512, 256, 64, 3]
        self.decoder_c = Decoder(ch_list=self.ch_list)
        self.decoder_base = Base_Decoder(ch_list=self.ch_list)
        self.classifier = Classifier(input_dim=classifier_input_dim, output_dim=classifier_output_dim)

    def forward(self, image_t, image_s):
        feature_t = self.encoder_base(image_t)
        feature_t_et = self.encoder_t(feature_t)
        feature_t_et_avg = self.encoder_t(feature_t, use_avg=True)
        feature_t_ec = self.encoder_c(feature_t)
        feature_t_ec_avg = self.encoder_c(feature_t, use_avg=True)
        #feature_t_ecdc = self.decoder_c(feature_t_ec_avg)
        feature_t_ = feature_t_ec + feature_t_et
        image_t_ = self.decoder_base(feature_t_)
        pred_t = self.classifier(feature_t_ec_avg)

        feature_s = self.encoder_base(image_s)
        feature_s_es = self.encoder_s(feature_s)
        feature_s_es_avg = (self.encoder_s(feature_s, use_avg=True))
        feature_s_ec = self.encoder_c(feature_s)
        feature_s_ec_avg = self.encoder_c(feature_s, use_avg=True)
        #feature_s_ecdc = self.decoder_c(feature_s_ec_avg)
        feature_s_ = feature_s_ec + feature_s_es
        image_s_ = self.decoder_base(feature_s_)
        pred_s = self.classifier(feature_s_ec_avg)
        
        return feature_t_et_avg, feature_t_ec_avg, image_t_, pred_t,\
                    feature_s_es_avg, feature_s_ec_avg, image_s_, pred_s