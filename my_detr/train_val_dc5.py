from model import build_detr_model
import paddle.optimizer as opt
import paddle
from coco_dataset import COCODetection, build as build_dataset
import math
import sys
import numpy as np
from coco_eval import CocoEvaluator
import os
import pickle

#字典转结构体
class DictToStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

#参数配置 COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
params = {"hidden_dim": 256, "lr_backbone": -1, "masks": False, "dilation": True, "backbone": "resnet50", "num_classes":91, 
    "dropout": 0.1, "nheads": 8, "dim_feedforward": 2048, "enc_layers": 6, "dec_layers": 6, "pre_norm": False,
    "num_queries": 100, "aux_loss": True, "set_cost_class": 1, "set_cost_bbox": 5,  "set_cost_giou": 2, "bbox_loss_coef": 5, 
    "giou_loss_coef": 2, "eos_coef":0.1, "coco_path": "/home/aistudio/data/data7122", "lr": 1e-6, "clip_max_norm": 0.1,
    "batch_size": 1, "epochs": 1} #batch可达12

args = DictToStruct(**params)

model, criterion, postprocesser = build_detr_model(args)
model_state_dict = paddle.load("./detr_dc5_save/best_model.pdparams")
model.set_state_dict(model_state_dict)

clip = paddle.nn.ClipGradByNorm(clip_norm=args.clip_max_norm)
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=200, gamma=0.8, verbose=True)
adam = opt.Adam(learning_rate=scheduler, grad_clip=clip, parameters=model.parameters())

#构造数据集
dataset_train = build_dataset(image_set="val", args=args)
train_loader = dataset_train.batch_reader(args.batch_size)
dataset_val = build_dataset(image_set="val", args=args)
val_loader = dataset_val.batch_reader(1) #暂时只能是1

#进行评估
@paddle.no_grad()
def evaluate():
    model.eval()
    coco_evaluator = CocoEvaluator(dataset_val.coco)
    for batch_id, (image, label)  in enumerate(val_loader()):
        outputs = model(image)
        orig_target_sizes = paddle.stack([t["orig_size"] for t in label], axis=0)
        results = postprocesser(outputs, orig_target_sizes)
        res = {target['im_id'].numpy()[0]: output for target, output in zip(label, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    return coco_evaluator.summarize() #只返回第一个评估指标

def train():
    best_map = 0
    for epoch_id in range(args.epochs):
        model.train()
        for batch_id, (image, label) in enumerate(train_loader()):
            out = model(image)
            loss_dict = criterion(out, label)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses = losses / args.batch_size

            if not math.isfinite(losses.numpy()):
                print("Loss is {}, stopping training".format(losses.numpy()))
                print(loss_dict)
                sys.exit(1)

            losses.backward()
            adam.minimize(losses)
            adam.clear_gradients()

            if batch_id % 100 ==0:
                print("epoch: {}, batch_id: {}, loss: {}".format( epoch_id, batch_id, np.mean(losses.numpy())))

        if epoch_id % 1 == 0:
            print("start evaluating....")
            tmp_map = evaluate()
            if tmp_map > best_map:
                print("save weights with map: ", tmp_map)
                paddle.save(model.state_dict(), "./detr_dc5_save/model.pdparams")
                paddle.save(adam.state_dict(), "./detr_dc5_save/model.pdopt")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("输入train或eval")
        exit(0)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "eval":
        evaluate()
    else:
        print("参数错误")
