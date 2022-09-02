# Global imports
import copy
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops


# Legacy resnet50 backbone
class OldBackbone(nn.Sequential):
    def __init__(self, resnet):
        super(OldBackbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(OldBackbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


# Legacy resnet50 head
class OldRes5Head(nn.Sequential):
    def __init__(self, resnet):
        super(OldRes5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(OldRes5Head, self).forward(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Generic Backbone
class Backbone(nn.Module):
    def forward(self, x):
        y = self.body(x)
        return y


# Generic Head
class Head(nn.Module):
    def forward(self, x):
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Resnet Backbone
class ResnetBackbone(Backbone):
    def __init__(self, resnet):
        super().__init__()
        return_layers = {
            'layer3': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.out_channels = 1024


# Resnet Head
class ResnetHead(Head):
    def __init__(self, resnet):
        super().__init__()
        self.head = resnet.layer4
        self.out_channels = [1024, 2048]
        self.featmap_names = ['feat_res4', 'feat_res5']


# Convnext Backbone
class ConvnextBackbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '5': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(convnext.features, return_layers=return_layers)
        self.out_channels = convnext.features[5][-1].block[5].out_features


# Convnext Head
class ConvnextHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.features[6],
            convnext.features[7],
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,
            convnext.features[7][-1].block[5].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']


# resnet model builder function
def build_resnet(arch='resnet50', pretrained=True,
        freeze_backbone_batchnorm=True, freeze_layer1=True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    # weights
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    # load model
    if freeze_backbone_batchnorm:
        resnet = torchvision.models.resnet50(weights=weights, norm_layer=norm_layer)
    else:
        resnet = torchvision.models.resnet50(weights=weights)

    # freeze first layers
    resnet.conv1.requires_grad_(False)
    resnet.bn1.requires_grad_(False)
    if freeze_layer1:
        resnet.layer1.requires_grad_(False)

    # setup backbone architecture
    backbone, head = ResnetBackbone(resnet), ResnetHead(resnet)
    
    # return backbone, head
    return backbone, head


# convnext model builder function
def build_convnext(arch='convnext_base', pretrained=True, freeze_layer1=True):
    # weights
    if pretrained:
        weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    else:
        weights = None

    # load model
    if arch == 'convnext_tiny':
        print('==> Backbone: ConvNext Tiny')
        convnext = torchvision.models.convnext_tiny(weights=weights)
    elif arch == 'convnext_small':
        print('==> Backbone: ConvNext Small')
        convnext = torchvision.models.convnext_small(weights=weights)
    elif arch == 'convnext_base':
        print('==> Backbone: ConvNext Base')
        convnext = torchvision.models.convnext_base(weights=weights)
    elif arch == 'convnext_large':
        print('==> Backbone: ConvNext Large')
        convnext = torchvision.models.convnext_large(weights=weights)
    else:
        raise NotImplementedError

    # freeze first layer
    if freeze_layer1:
        convnext.features[0].requires_grad_(False)

    # setup backbone architecture
    backbone, head = ConvnextBackbone(convnext), ConvnextHead(convnext)
    
    # return backbone, head
    return backbone, head
