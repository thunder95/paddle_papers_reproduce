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

#注意，必须用32G显存，否则显存不够, 而且显存占用也不太固定， torch上验证出0.420
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420

#字典转结构体
class DictToStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

#参数配置 COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
params = {"hidden_dim": 256, "lr_backbone": -1, "masks": False, "dilation": False, "backbone": "resnet50", "num_classes":91,
    "dropout": 0.1, "nheads": 8, "dim_feedforward": 2048, "enc_layers": 6, "dec_layers": 6, "pre_norm": False,
    "num_queries": 100, "aux_loss": True, "set_cost_class": 1, "set_cost_bbox": 5,  "set_cost_giou": 2, "bbox_loss_coef": 5, 
    "giou_loss_coef": 2, "eos_coef":0.1, "coco_path": "/f/dataset/COCO2017", "lr": 1e-6, "clip_max_norm": 0.1,
    "batch_size": 1, "epochs": 10} #batch可达12

args = DictToStruct(**params)

model, criterion, postprocesser = build_detr_model(args)

# with open('/home/aistudio/data/data88770/weights/wts_dc5.pkl', 'rb') as f:
#     pkl_wts = pickle.load(f)
#     print(len(pkl_wts.keys()))
# model_state_dict = paddle.load("./detr_dc5_save/model.pdparams")
# for k in model_state_dict:
#     if model_state_dict[k].shape != pkl_wts[k].shape:
#         print("===>", k, pkl_wts[k], model_state_dict[k].shape, pkl_wts[num].shape)
#         continue
#     model_state_dict[k] = pkl_wts[k]
# model.set_state_dict(model_state_dict) #设置时不匹配， backbone.0.body.layer4.2.bn3 [256, 256] > [2048], 但是但是上面校验也是100%匹配

# old_model_vals = list(pkl_wts.values())
# old_model_keys = list(pkl_wts.keys())
# print()
# new_model_keys = list(model_state_dict.keys())
# old_set = set(old_model_keys)
# new_set = set(new_model_keys)
# print(list(set(new_set) - (set(old_set))))
# print(list(set(old_set) - (set(new_set))))
# print(len(list(set(old_set) - (set(new_set)))))

#加载预训练模型
# num = 0
# for k in model_state_dict:
#     if model_state_dict[k].shape != old_model_vals[num].shape:
#         print("===>", k, old_model_keys[num], model_state_dict[k].shape, old_model_vals[num].shape)
#         continue
#     model_state_dict[k] = old_model_vals[num]
#     num += 1

# oks = list(old_model_dict.keys())
# for k in range(len(oks)):
#     if k < 50:
#         print(oks[k])


# old_set = set(old_model_dict.keys())
# print(len(old_set))

# new_set = set(mdoel_state_dict.keys())
# print(len(new_set))
# print(list(set(new_set) - (set(old_set)))[0])
# print(list(set(old_set) - (set(new_set)))[0])
# print(len(list(set(old_set) - (set(new_set)))))

# check_list = []
# for x in new_set:
#     if "layer1.2" in x:
#         check_list.append(x)
# print(check_list)
# check_list = []
# for x in old_set:
#     if "layer1.2" in x:
#         check_list.append(x)
# print(check_list)
# exit()]
#print(mdoel_state_dict["backbone.0.body.layer4.2.bn3._variance"].shape)
# model.set_state_dict(mdoel_state_dict) #设置时不匹配， backbone.0.body.layer4.2.bn3 [256, 256] > [2048], 但是但是上面校验也是100%匹配
# exit()

clip = paddle.nn.ClipGradByNorm(clip_norm=args.clip_max_norm)
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=200, gamma=0.8, verbose=True)
adam = opt.Adam(learning_rate=scheduler, grad_clip=clip, parameters=model.parameters())

#构造数据集
# dataset_train = build_dataset(image_set="train", args=args)
# train_loader = dataset_train.batch_reader(args.batch_size)
dataset_val = build_dataset(image_set="val", args=args)
val_loader = dataset_val.batch_reader(1) #暂时只能是1

# train_loader = paddle.io.DataLoader(dataset_train,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True)

# val_loader = paddle.io.DataLoader(dataset_val,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=False)

#进行评估
@paddle.no_grad()
def evaluate():
    model.eval()
    coco_evaluator = CocoEvaluator(dataset_val.coco)
    for batch_id, (image, label) in enumerate(val_loader()):
        print(batch_id)
        # print("image.shape", image[0].shape, type(image[0]))
        # print("label.shape===>", label[0])
        # image = [paddle.ones(shape=[3, 800, 1199])]
        # print(label) #输出的是相对位置坐标
        # exit()
        print(len(label))
        outputs = model(image)
    #     # print("model output: ", outputs)
    #     # exit()
    #     # print(outputs.keys(), len(label), label[0].keys())
    #     orig_target_sizes = paddle.stack([t["orig_size"] for t in label], axis=0)
    #     # print(orig_target_sizes)
    #     results = postprocesser(outputs, orig_target_sizes)

    #     res = {target['im_id'].numpy()[0]: output for target, output in zip(label, results)}
    #     coco_evaluator.update(res)
    #     # print("results", results)

    # coco_evaluator.synchronize_between_processes()
    # coco_evaluator.accumulate()
    # return coco_evaluator.summarize() #只返回第一个评估指标

evaluate() #只是验证
exit()

# print(model)
# exit()

def train():
    best_map = 0
    for epoch_id in range(args.epochs):
        model.train()
        for batch_id, (image, label) in enumerate(train_loader()):
           
            # image = [paddle.ones(shape=[3, 800, 1199])]
            out = model(image)
            
            # print("out: ", out['pred_logits']) #checked
            # print(out)

            #保存模型权重名称
            print("model===>", model)
            paddle.save(model.state_dict(), "./detr_dc5_save/model.pdparams")
            paddle.save(adam.state_dict(), "./detr_dc5_save/model.pdopt")
            exit()

            loss_dict = criterion(out, label)
            # for k in loss_dict:
            #     print(k, loss_dict[k])        
             
            
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

            if batch_id % 10 ==0:
                print("epoch: {}, batch_id: {}, loss: {}".format( epoch_id, batch_id, np.mean(losses.numpy())))

        if epoch_id % 1 == 0:
            print("start evaluating....")
            tmp_map = evaluate()
            if tmp_map > best_map:
                print("save weights with map: ", tmp_map)
                paddle.save(model.state_dict(), "./dla_param_save/model.pdparams")
                paddle.save(adam.state_dict(), "./dla_param_save/model.pdopt")

#开始训练
train()


        #     print("===>", model.state_dict().keys())
        #     exit()

    # if epoch_id % 2 ==0:
    #     model.eval()
    #     batch_acc = []
    #     for batch_id, (image, label)  in enumerate(val_loader()):
    #         out = model(image)
    #         eval_acc = paddle.metric.accuracy(out, label)
    #         batch_acc.append(eval_acc.numpy())

    #     mean_acc = np.mean(np.asarray(batch_acc))
    #     print("mean acc: ", mean_acc, best_acc)
    #     if mean_acc > best_acc:
    #         paddle.save(model.state_dict(), "./param_save/new_model.pdparams")
    #         paddle.save(adam.state_dict(), "./param_save/new_model.pdopt")
