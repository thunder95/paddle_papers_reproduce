import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Conv3D, Dropout


#加一个simple_consensus算子，没必要，直接用reduce_mean
# def simple_consensus_forward(x):
#     print(x)
#     return x
# def simple_consensus_backward(x):
#     print(x)
#     return x
# def create_tmp_var(program, name, dtype, shape):
#     return program.current_block().create_var(name=name, dtype=dtype, shape=shape)


# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer3D(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=(0,0,0),
                 act=None):
        super(ConvBNLayer3D, self).__init__(name_scope)

        # 创建卷积层
        self._conv = Conv3D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
"""
depth: 3*4*6*3
block1:
    stride: 1*1*1, pad: 0 , kernel: 1*1*1
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3
    stride: 1*1*1, pad: 0 , kernel: 1*1*1 【没有relu】

    stride: 1*1*1, pad: 0 , kernel: 1*1*1
    num_filters全是64

    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【输入*4】
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=64
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=64 【没有relu】

    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【输入*4】
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=64
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=64 【没有relu】

    【add,此处有checkpoint， 输出256】
    【relu】 【没有nonlocal】

block2:
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=128
    stride: 1*2*2, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=128
    stride: 1*1*1, pad: 0 , kernel: 1*1*1 , num_filters=128 【没有relu】

    stride: 1*2*2, pad: 0 , kernel: 1*1*1, num_filters=256

    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=512  【输入*4】
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=128  
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=128 【没有relu】

    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=512  【输入*4】
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=128  
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=128 【没有relu】

    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=512  【输入*4】
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=128  
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=128 【没有relu】

    【add,此处有checkpoint， 输出512】
    【relu】 【没有nonlocal】

block3:(注意inflate方式不同)
    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=512
    stride: 1*2*2, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256
    stride: 1*1*1, pad: 0 , kernel: 1*1*1 , num_filters=256 【没有relu】

    stride: 1*2*2, pad: 0 , kernel: 1*1*1, num_filters=512

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【没有relu】

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【没有relu】

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【没有relu】

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【没有relu】

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=256  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=256 【没有relu】

    【add,此处有checkpoint， 输出1024】
    【relu】 【没有nonlocal】

block4:(注意inflate方式不同)
    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=1024
    stride: 1*2*2, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=512
    stride: 1*1*1, pad: 0 , kernel: 1*1*1 , num_filters=512 【没有relu】

    stride: 1*2*2, pad: 0 , kernel: 1*1*1, num_filters=1024

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=2048  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=512  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=512 【没有relu】

    stride: 1*1*1, pad: 1, 0, 0 , kernel: 3*1*1, num_filters=2048  
    stride: 1*1*1, pad: 0, 1, 1 , kernel: 1*3*3, num_filters=512  【除以*4】
    stride: 1*1*1, pad: 0 , kernel: 1*1*1, num_filters=512 【没有relu】
    【add,此处有checkpoint， 输出1024】
    【relu】 【没有nonlocal】
"""

#带分支的block， 注意有inflate
class BottleneckBlock3D_A(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels, #输入通道数
                 num_filters,
                 stride_2,
                 filter_1,
                 pad_1):
        super(BottleneckBlock3D_A, self).__init__(name_scope)

           # 创建第一个卷积层 1x1x1
        self.conv0 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=(filter_1, 1, 1),
            stride = 1,
            padding= (pad_1, 0, 0),
            act='relu')

        # 创建第二个卷积层 1x3x3
        self.conv1 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=(1, 3, 3),
            stride=(1, stride_2, stride_2),
            padding= (0, 1, 1),
            act='relu')

        # 创建第三个卷积 1x1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            stride=1,
            padding= 0,
            act=None)

        self._num_channels_out = num_filters * 4

        #short_cut卷积层
        self.short = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters * 4,
            filter_size=1,
            stride=(1, stride_2, stride_2),
            padding= 0,
            act=None)

    def forward(self, inputs):
        y = self.conv0(inputs)
        # print(1, y.shape)
        y = self.conv1(y)
        # print(2, y.shape)
        y = self.conv2(y)
        # print(3, y.shape)

        #shortcut_conv
        inputs = self.short(inputs)
        # print(4, inputs.shape)
        y = fluid.layers.elementwise_add(x=inputs, y=y)
        # print(5, y.shape)
        return fluid.layers.relu(y)

#不带分支的block，只依赖于输入维度，结构比较统一
class BottleneckBlock3D_B(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters):
        super(BottleneckBlock3D_B, self).__init__(name_scope)

        # 创建第一个卷积层 1x1x1
        self.conv0 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            stride = 1,
            padding= 0,
            act='relu')

        # 创建第二个卷积层 1x3x3
        self.conv1 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=(1, 3, 3),
            stride=1,
            padding= (0, 1, 1),
            act='relu')

        # 创建第三个卷积 1x1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer3D(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            stride=1,
            padding= 0,
            act=None)

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        y = self.conv1(y)
        y = self.conv2(y)
        
        y = fluid.layers.elementwise_add(x=inputs, y=y)
        return fluid.layers.relu(y)


class ResNet3D(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50):
        """
        name_scope，模块名称
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet3D, self).__init__(name_scope)
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        self.conv1_kernel_t = 1
        self.conv1_stride_t = 1
        self.pool1_kernel_t = 1
        self.pool1_stride_t = 1


        # 残差块中使用到的卷积的输出通道数
        strides_2 = (1, 2, 2, 2)
        kernels_1 = (1, 1, 3, 3)
        paddings_1 = (0, 0, 1, 1)
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer3D(
            self.full_name(),
            num_channels=3,
            num_filters=64,
            filter_size=(self.conv1_kernel_t, 7, 7),
            stride=(self.conv1_stride_t, 2, 2),
            padding=((self.conv1_kernel_t - 1) // 2, 3, 3),
            act="relu")

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        self.indice_3rd = 0
        for block in range(len(depth)):
            #第一个
            bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, 0),
                    BottleneckBlock3D_A(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride_2=strides_2[block],
                        filter_1=kernels_1[block],
                        pad_1 = paddings_1[block]))
            num_channels = bottleneck_block._num_channels_out
            self.bottleneck_block_list.append(bottleneck_block)


            #其他
            for i in range(1, depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock3D_B(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        ))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)

            if block == 2:
                self.indice_3rd = len(self.bottleneck_block_list) - 1

        #==================RESNET3D构造结束==========================#

    def forward(self, inputs):
        y = self.conv(inputs)
        y = fluid.layers.pool3d(
              input=y, # shape: [2, 3, 8, 8, 8]
              pool_size=[self.pool1_kernel_t,3,3],
              pool_type='max',
              pool_stride=(self.pool1_stride_t, 2, 2),
              pool_padding=(self.pool1_kernel_t // 2, 1, 1))
        
        num = 0
        y1 = None
        for bottleneck_block in self.bottleneck_block_list:
            # print("num: ", num)
            y = bottleneck_block(y)
            if num == self.indice_3rd:
                y1 = y
            num+=1

        # print("res3d y: ", y.shape)
        # print("3rd y: ", y1.shape, self.indice_3rd)
        return y1, y

#resnet3d倒数第二层进行分类
class AuxHead(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, class_dim=1, dropout_ratio=0.5):
        super(AuxHead, self).__init__(name_scope)

        #aux_head, 权重0.5, 只影响损失值, 在外层使用交叉熵
        self.conv = ConvBNLayer3D(
            self.full_name(),
            num_channels = num_channels, #1024
            num_filters = num_channels * 2, #2048
            filter_size = (1, 3, 3),
            stride = (1, 2, 2),
            padding = (0, 1, 1),
            act = "relu"
        )
        self.dropout = fluid.dygraph.Dropout(p=dropout_ratio)
        self.fc = Linear(num_channels*2, class_dim)

    def forward(self, inputs):
        x = self.conv(inputs)
        # print(x.shape)
        #这里需要填pool_size， 也就是最终输出大小(N,C,poolsize[0],poolsize[1],poolsize[2])
        # torch: the target output size (single integer or triple-integer tuple)
        x = fluid.layers.adaptive_pool3d(
                  input=x,
                  pool_size=[1, 1, 1],
                  pool_type='avg') #输出应该是1*1*1，相当于全局池化
        # print("pool shape", x.shape)

        x = fluid.layers.reshape(x=x, shape=[-1, 2048])
        x = self.dropout(x)
        x = self.fc(x)
        return x

#空间模块
"""
ds_num有2个值， 1， 0
convmodule只遍历了一个值0
所以执行的只有一次convmodule和一次identity
inputs有两个值
    input[0] -> convmodule
    input[1] -> identity(没有任何操作直接返回)

"""


#时域模块
class TemporalModulation(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(TemporalModulation, self).__init__(name_scope)

        #输入2048， 输出1024
        self.conv = ConvBNLayer3D(
            self.full_name(),
            num_channels = 2048,
            num_filters = 1024,
            filter_size = (3, 1, 1),
            stride = (1, 1, 1),
            padding = (1, 0, 0),
            act = "relu",
            groups = 32
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        x = fluid.layers.pool3d(
              input=x, 
              pool_size=[32, 1, 1],
              pool_type='max',
              pool_stride=(32, 1, 1),
              pool_padding=(0, 0, 0),
              ceil_mode=True)
        return x

#下采样融合模块
class LevelFusion(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(LevelFusion, self).__init__(name_scope)

        #输入输出都是1024， 最终融合到2048
        self.conv_x1 = ConvBNLayer3D(
            self.full_name(),
            num_channels = 1024,
            num_filters = 1024,
            filter_size = (1, 1, 1),
            stride = (1, 1, 1),
            padding = (0, 0, 0),
            act = "relu",
            groups = 32
        )

        self.conv_y1 = ConvBNLayer3D(
            self.full_name(),
            num_channels = 1024,
            num_filters = 1024,
            filter_size = (1, 1, 1),
            stride = (1, 1, 1),
            padding = (0, 0, 0),
            act = "relu",
            groups = 32
        )

        self.conv_out = ConvBNLayer3D(
            self.full_name(),
            num_channels = 2048,
            num_filters = 2048,
            filter_size = (1, 1, 1),
            stride = (1, 1, 1),
            padding = (0, 0, 0),
            act = "relu",
            groups = 1
        )

    def forward(self, input_x, input_y):
        x1 = fluid.layers.pool3d(
              input=input_x, 
              pool_size=[1, 1, 1],
              pool_type='max',
              pool_stride=(1, 1, 1),
              pool_padding=(0, 0, 0),
              ceil_mode=True)
        x1 = self.conv_x1(x1)

        y1 = fluid.layers.pool3d(
              input=input_y, 
              pool_size=[1, 1, 1],
              pool_type='max',
              pool_stride=(1, 1, 1),
              pool_padding=(0, 0, 0),
              ceil_mode=True)
        y1 = self.conv_y1(y1)

        # print("===>", x1.shape, y1.shape)
        out = fluid.layers.concat(input=[x1,y1], axis=1)
        # print(out.shape)
        out = self.conv_out(out)
        return out

#TPN主干网络
class TPN(fluid.dygraph.Layer):
    #test模式下num_seg = 10(config) * 3
    def __init__(self, name_scope, layers=50, class_dim=1, num_seg = 1):
        super(TPN, self).__init__(name_scope)
        self.num_seg = num_seg

        self.resnet3d = ResNet3D(self.full_name(), 50)
        self.aux_head = AuxHead(self.full_name(), num_channels=1024, class_dim=101, dropout_ratio=0.5)

        #空间模块1(input[0]->convmodule)
        self.spatial_module = ConvBNLayer3D(
            self.full_name(),
            num_channels = 1024,
            num_filters = 2048,
            filter_size = (1, 3, 3),
            stride = (1, 2, 2),
            padding = (0, 1, 1),
            act = "relu"
        )

        #时域模块
        self.temporal_module_1 = TemporalModulation(self.full_name())
        self.temporal_module_2 = TemporalModulation(self.full_name())

        #左侧上采样融合
        self.level_fusion_op2 = LevelFusion(self.full_name())

        #下采样
        self.down_conv = Conv3D(
            num_channels=1024,
            num_filters=1024,
            filter_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            groups=1,
            act=None,
            bias_attr=False)

        #右侧下采样融合
        self.level_fusion_op1 = LevelFusion(self.full_name())

        #融合后的convbn
        self.fuse_conv = ConvBNLayer3D(
            self.full_name(),
            num_channels=4096,
            num_filters=2048,
            filter_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            groups=1,
            act="relu")

        #dropout
        self.dropout_head = fluid.dygraph.Dropout(p=0.5)
        self.cls_out = Linear(input_dim=2048, output_dim=class_dim, act='softmax')

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, shape=[-1, 3, 32, 224, 224]) #seglen=32
        y1, y2 = self.resnet3d(inputs)
        # print(y1.shape, y2.shape)
        aux_head_output = self.aux_head(y1) #最终输出，融合的loss
        y1 = self.spatial_module(y1) #空间模块

        #时域模块
        y1 = self.temporal_module_1(y1) #图的左边
        y2 = self.temporal_module_2(y2) #图的中间(右边)
        # temporal_modulation_outs = [y1, y2] #暂存 ====>

        #y2上采样并融合y1
        tmp_y2 =  fluid.layers.reshape(x=y2, shape=[-1, y2.shape[2], y2.shape[3], y2.shape[4]])
        tmp_y2 = fluid.layers.interpolate(input=tmp_y2, scale=1, resample ='NEAREST') #不支持(1, 1, 1)格式
        tmp_y2 = fluid.layers.reshape(x=tmp_y2, shape=[-1, 1024, y2.shape[2], y2.shape[3], y2.shape[4]])
        y1 = fluid.layers.elementwise_add(x=y1, y=tmp_y2) #====>最中间核心的ADD

        topdownouts = self.level_fusion_op2(y1, y2) #左边： y1融合后，再和时域y2进行融合


        #y2融合y1的下采样,成为y3
        tmp_y1_down = self.down_conv(y1)
        tmp_y1_down = fluid.layers.pool3d(
              input=tmp_y1_down, 
              pool_size=[1, 1, 1],
              pool_type='max',
              pool_stride=(1, 1, 1),
              pool_padding=(0, 0, 0),
              ceil_mode=True)
        y3 = fluid.layers.elementwise_add(x=y2, y=tmp_y1_down)

        #下采样融合y1, y3
        y4 = self.level_fusion_op1(y1, y3) #左边： y1和下采样后的y3进行融合
        # print("y4 shape: ", y4.shape)

        #合并y4和topdownouts
        y5 = fluid.layers.concat(input=[topdownouts, y4], axis=1)
        y5 = self.fuse_conv(y5)
        # print("y5 shape: ", y5.shape)

        #SimpleSpatialTemporalModule pool3d
        y6 = fluid.layers.pool3d(
              input=y5, 
              pool_size=[1, 7, 7],
              pool_type='avg',
              pool_stride=(1, 1, 1),
              pool_padding=(0, 0, 0),
              ceil_mode=True)
        # print("y6 shape: ", y6.shape, y6.shape[1:])

        #consensus
        #expand_dims -1
        # out_var = create_tmp_var(fluid.default_main_program(), name='output_op', dtype='float32', shape=y6.shape)
        # fluid.layers.py_func(func=simple_consensus_forward, x=y6, out=out_var, 
        #     backward_func=simple_consensus_backward, skip_vars_in_backward_input=y6)
        
        #测试的时候就是 segnum=10， 经过three_crop之后就是30， 如果是train和valid就是1

        y6 = fluid.layers.reshape(x=y6, shape=[-1, self.num_seg, y6.shape[1], y6.shape[2], y6.shape[3]])
        y6 = fluid.layers.reduce_mean(y6, dim=1, keep_dim=True) 
        # print("y6 shape: ", y6.shape)

        #cls_head
        y6 = fluid.layers.reshape(x=y6, shape=[-1, 2048])
        # print("y6 shape: ", y6.shape)

        y6 = self.dropout_head(y6)
        out = self.cls_out(y6)

        return aux_head_output, out

#测试网络结构
if __name__ == '__main__':
    # 测试resnet3d
    # with fluid.dygraph.guard():
    #     network = ResNet3D("resnet3d", 50, 101)
    #     img = np.zeros([1, 3, 32, 224, 224]).astype('float32')
    #     img = fluid.dygraph.to_variable(img)
    #     outs = network(img)

    # 测试主干网络tpn
    with fluid.dygraph.guard():
        network = TPN("tpn", 50, 101)
        img = np.zeros([3, 3, 32, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img)
        print(outs[1].numpy())
    print("done")
