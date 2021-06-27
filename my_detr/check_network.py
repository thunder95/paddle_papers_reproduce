import argparse
import numpy as np
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.argument import get_args_parser
from backbone import build_backbone
from transformer import build_transformer
import paddle
from model import build_detr_model, HungarianMatcher

if __name__ == '__main__':
    #模拟参数
    class DictToStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    params = {"hidden_dim": 256, "lr_backbone": -1, "masks": False, "dilation": False, "backbone": "resnet50", "num_classes":91, 
        "hidden_dim": 256, "dropout": 0.1, "nheads": 8, "dim_feedforward": 2048, "enc_layers": 6, "dec_layers": 6, "pre_norm": False,
        "num_queries": 100, "aux_loss": True, "set_cost_class": 1, "set_cost_bbox": 5,  "set_cost_giou": 2, "bbox_loss_coef": 5, 
        "giou_loss_coef": 2, "eos_coef":0.1}
    args = DictToStruct(**params)
    
    backbone = build_backbone(args)
    fake_image = paddle.zeros(shape=[4, 3, 512, 512], dtype='float32')
    mask = paddle.zeros(shape=[4, 512, 512], dtype='bool')
    fake_data = NestedTensor(fake_image, mask)
    for k, v in backbone.state_dict().items():
        print(k + ': ' + str(v.shape))

    out, pos = backbone(fake_data)



    for feature_map in out:
        print(feature_map.tensors.shape) # [4, 2048, 16, 16]
        print(feature_map.mask.shape) # [4, 16, 16]

    for pos_tensor in pos:
        print(pos_tensor.shape) # [4, 256, 16, 16]
        
    transformer = build_transformer(args)
    features = paddle.zeros(shape=[4, 256, 16, 16], dtype='float32')
    mask = paddle.zeros(shape=[4, 16, 16], dtype='bool')
    query_embed = paddle.zeros(shape=[100, 256], dtype='float32')
    pos_embed = paddle.zeros(shape=[4, 256, 16, 16], dtype='float32')
    hs, memory = transformer(features, mask, query_embed, pos_embed)
    print(hs.shape) # [6, 4, 100, 256]
    print(memory.shape) # [4, 256, 16, 16]
        
    detr, criterion, postprocessors = build_detr_model(args)
    out = detr(fake_data)
    for name, tensor in out.items():
        if isinstance(tensor, list):
            print (name)
            print()
            for aux_loss in tensor:
                for name, tensor in aux_loss.items():
                    print(name)
                    print(tensor.shape)
        else:
            print(name)
            print(tensor.shape)

    # {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4],
    #  "aux_outputs": [
    #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
    #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
    #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
    #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
    #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
    # ]} OK

    # for k, v in detr.state_dict().items():
    #     print(k + ": " + str(v.shape))

    target = [
        {"labels": paddle.zeros(shape=[6, ], dtype='float32'),
            "boxes": paddle.zeros(shape=[6, 4], dtype='float32')},
        {"labels": paddle.zeros(shape=[3, ], dtype='float32'),
            "boxes": paddle.zeros(shape=[3, 4], dtype='float32')},
        {"labels": paddle.zeros(shape=[17, ], dtype='float32'),
            "boxes": paddle.zeros(shape=[17, 4], dtype='float32')},
        {"labels": paddle.zeros(shape=[5, ], dtype='float32'),
            "boxes": paddle.zeros(shape=[5, 4], dtype='float32')},
    ]
    
    matcher = HungarianMatcher(1, 1, 1)
    indices = matcher(out, target)
    for ind in indices:
        i_ind, j_ind = ind
        print(i_ind.shape, j_ind.shape)
        # [6] [6]
        # [3] [3]
        # [17] [17]
        # [5] [5]

    loss = criterion(out, target)
    for name, loss in loss.items():
        print(name)
        print(loss) #OK

# ‘‘‘
#  [0.76757371])
#  [0.])
#    [2.])
#    [392.25000000])
# ’’’






