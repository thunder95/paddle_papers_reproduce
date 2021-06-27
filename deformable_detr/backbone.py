# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from collections import OrderedDict
from typing import Dict, List
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import resnet
import math

from intermediate_layer_getter import IntermediateLayerGetter
from util.misc import NestedTensor

import paddle.vision.ops.DeformConv2D

class FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.module = nn.BatchNorm(n, use_global_stats=True, param_attr=paddle.ParamAttr(learning_rate=0))
    
    def forward(self, x):
        return self.module(x)

class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passes")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        bs, h, w = mask.shape

        mask = mask.numpy()
        not_mask = ~mask
        not_mask = paddle.Tensor(not_mask).astype('float32')
        y_embed = paddle.cumsum(not_mask, axis=1) # [batch_size, h, w]
        x_embed = paddle.cumsum(not_mask, axis=2) # [batch_size, h, w]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = (np.arange(0, self.num_pos_feats, 1, dtype="float32")) # [num_pos_feats]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [num_pos_feats]
        dim_t = paddle.to_tensor(dim_t)

        x_embed = paddle.unsqueeze(x_embed, 3) # [batch_size, h, w, 1]
        y_embed = paddle.unsqueeze(y_embed, 3) # [batch_size, h, w, 1]
        pos_x = x_embed / dim_t           # [batch_size, h, w, num_pos_feats]
        pos_y = y_embed / dim_t           # [batch_size, h, w, num_pos_feats]
        pos_x_1 = paddle.sin(pos_x[:, :, :, 0::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_x_2 = paddle.cos(pos_x[:, :, :, 1::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_y_1 = paddle.sin(pos_y[:, :, :, 0::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_y_2 = paddle.cos(pos_y[:, :, :, 1::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_x = paddle.reshape(paddle.stack([pos_x_1, pos_x_2], axis=4), (bs, h, w, -1)) # [batch_size, h, w, num_pos_feats]
        pos_y = paddle.reshape(paddle.stack([pos_y_1, pos_y_2], axis=4), (bs, h, w, -1)) # [batch_size, h, w, num_pos_feats]

        pos = paddle.concat((pos_y, pos_x), axis=3)    # [batch_size, h, w, num_pos_feats * 2]
        pos = paddle.transpose(pos, perm=(0, 3, 1, 2)) # [batch_size, num_pos_feats * 2, h, w]
        return pos


class BackboneBase(nn.Layer):

    def __init__(self, backbone: nn.Layer, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            #优先级为是not>and>or，
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.stop_gradient = True

        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        return xs

        #暂时没用到分割
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     m = paddle.unsqueeze(m, 1) # [batch_size, h, w] -> [batch_size, 1, h, w]
        #     m = m.astype("float32")
        #     mask = F.interpolate(m, size=x.shape[-2:]) #resize
        #     mask = mask.astype("bool")
        #     mask = paddle.squeeze(mask, [1]) # [batch_size, 1, h, w] -> [batch_size, h, w]
        #     out[name] = NestedTensor(x, mask)
        #
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 num_classes: int,
                 dilation: bool):
        if dilation:
            backbone = resnet.resnet50(num_classes=num_classes, replace_stride_with_dilation=[False, False, True], norm_layer=FrozenBatchNorm2d) #DC5
        else:
            backbone = resnet.resnet50(num_classes=num_classes, norm_layer=FrozenBatchNorm2d) #普通的resnet50

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x))

        return out, pos

def build_backbone(args):
    # print("---->")
    # print(args.__dict__)
   
    N_steps = args.hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    train_backbone = args.lr_backbone > 0 #是否训练backbone
    return_interm_layers = args.masks or (args.num_feature_levels > 1) #这里是TRUE

    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

if __name__ == '__main__':
    #字典转结构体
    class DictToStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    #COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    params = {"hidden_dim": 256, "lr_backbone": -1, "masks": False, "dilation": True, "backbone": "resnet50", "num_classes":91}
    args = DictToStruct(**params)
    backbone = build_backbone(args)
    model = paddle.Model(backbone)
    # model = paddle.Model(res50)
    print(model.summary((1, 3, 224, 224)))

