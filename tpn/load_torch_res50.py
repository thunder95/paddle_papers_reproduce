import torch
import paddle
import paddle.fluid as fluid
from collections import OrderedDict
from tpn_model import TPN
import numpy as np

#建立一个映射表, 要花很多时间，建议后面做个工具
# dict_table = {
#     'conv1': 'resnet3d.conv._conv',
#     'bn1': 'resnet3d.conv._batch_norm',
#     'layer1.0.conv1':'resnet3d.bb_0_0.conv0._conv',
# }


torch_weight = torch.load('/home/hl/.cache/torch/checkpoints/resnet50-19c8e357.pth', map_location=torch.device("cpu"))
tk = []
for torch_key in torch_weight.keys():
    # print(torch_key)
    tk.append(torch_key)


print(tk[:20])
print("======================================")
pk=[]
with fluid.dygraph.guard():
    paddle_model = TPN("tpn", 50, 101)
    paddle_weight = paddle_model.state_dict()
    for paddle_key in paddle_weight:
        pk.append(paddle_key)
    print(pk)

    print("======================================")
    new_weight_dict = OrderedDict()

    for torch_key, paddle_key in zip(torch_weight.keys(), paddle_weight.keys()):
        if not torch_key.find('fc'):
            continue
            tmp_weight = torch_weight[torch_key].detach().numpy() #fc
            new_weight_dict[paddle_key] = tmp_weight

        else:
            tmp_weight = torch_weight[torch_key].detach().numpy().T
            print(torch_key, paddle_key, tmp_weight.shape, paddle_weight[paddle_key].shape)

            if len(tmp_weight.shape) == 4:
                tmp_weight = np.expand_dims(tmp_weight, axis=0).transpose((4, 3, 0, 1, 2))

            if (list(tmp_weight.shape) != paddle_weight[paddle_key].shape):
                print("what a fucking life....", torch_key, paddle_key, tmp_weight.shape, paddle_weight[paddle_key].shape )
                tmp_weight = tmp_weight.repeat(3, axis=2)
                tmp_weight = tmp_weight / 3.0
                print("redim: ", tmp_weight.shape)
            # print("===>", tmp_weight.shape)
            new_weight_dict[paddle_key] = tmp_weight

    print("weight added done")
    # print(len(new_weight_dict.keys()))


    #其他层的参数
    for paddle_key in paddle_weight.keys():
        if paddle_key not in new_weight_dict:
            print("not exists: ", paddle_key)
            new_weight_dict[paddle_key] = paddle_weight[paddle_key]
    paddle_model.set_dict(new_weight_dict)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(), 'pretrained.pdparams')




