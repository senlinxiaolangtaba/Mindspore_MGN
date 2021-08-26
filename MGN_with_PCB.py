import copy
import math
import os
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.ops import Squeeze
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from scipy.stats import truncnorm
from mindspore import ops
from resnet import kaiming_normal, Bottleneck
# from loss import MGN_Loss
import sys


#
# from resnet import resnet50


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                         padding=1, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                         padding=0, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel,
                         kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    else:
        weight_shape = (out_channel, in_channel)
        weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn_last(out_channel)
        if self.se_block:
            self.se_global_pool = P.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = P.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
                                                                         stride, use_se=self.use_se), _bn(out_channel)])
                else:
                    self.down_sample_layer = nn.SequentialCell([nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                                                                _conv1x1(in_channel, out_channel, 1,
                                                                         use_se=self.use_se), _bn(out_channel)])
            else:
                self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                     use_se=self.use_se), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = F.reshape(out, F.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualBlock_layer4(nn.Cell):  # yaogai
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock_layer4, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=1, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn_last(out_channel)
        if self.se_block:
            self.se_global_pool = P.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = P.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
                                                                         stride, use_se=self.use_se), _bn(out_channel)])
                else:
                    self.down_sample_layer = nn.SequentialCell([nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                                                                _conv1x1(in_channel, out_channel, 1,
                                                                         use_se=self.use_se), _bn(out_channel)])
            else:
                self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                     use_se=self.use_se), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = F.reshape(out, F.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualBlockBase(nn.Cell):

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False,
                 se_block=False,
                 res_base=True):
        super(ResidualBlockBase, self).__init__()
        self.res_base = res_base
        self.conv1 = _conv3x3(in_channel, out_channel, stride=stride, res_base=self.res_base)
        self.bn1d = _bn(out_channel)
        self.conv2 = _conv3x3(out_channel, out_channel, stride=1, res_base=self.res_base)
        self.bn2d = _bn(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True

        self.down_sample_layer = None
        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                 use_se=use_se, res_base=self.res_base),
                                                        _bn(out_channel, res_base)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1d(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2d(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 use_se=False,
                 res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = P.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer4 = self._make_layer_l4(block,
                                          layer_nums[3],
                                          in_channel=in_channels[3],
                                          out_channel=out_channels[3],
                                          stride=strides[3],
                                          use_se=self.use_se,
                                          se_block=self.se_block)

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def _make_layer_l4(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=1, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def se_resnet50(class_num=1001):
    """
    Get SE-ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of SE-ResNet50 neural network.

    Examples:
      net = se-resnet50(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  use_se=True)


def resnet101(class_num=1001):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
       net = resnet101(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


class ClassBlock(nn.Cell):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=256, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            # 写一下如何初始胡的参数
            weight_shape = (num_bottleneck, input_dim)
            weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
            add_block += [nn.Dense(input_dim, num_bottleneck, has_bias=True, weight_init=weight, bias_init=0)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            # 写一下如何初始胡的参数

            # weight_shape = (num_bottleneck, num_bottleneck)
            weight = Tensor(np.random.normal(
                1.0, 0.02, num_bottleneck).astype("float32"))
            # 就是这里不对
            add_block += [nn.BatchNorm1d(num_bottleneck, gamma_init=weight, beta_init='zero')]
            #     elif classname.find('BatchNorm1d') != -1:
            #         init.normal_(m.weight.data, 1.0, 0.02)
            #         init.constant_(m.bias.data, 0.0)
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(keep_prob=droprate)]
        add_block = nn.SequentialCell(*add_block)

        # add_block.apply(weights_init_kaiming)#要加入参数初始化的部分 dan没有apply这个方法，这里不对劲 要修改！！！！

        classifier = []
        # 写一下如何初始胡的参数
        weight_shape = (class_num, num_bottleneck)
        weight = Tensor(np.random.normal(
            0, 0.001, weight_shape).astype("float32"))
        classifier += [nn.Dense(num_bottleneck, class_num, weight_init=weight, bias_init=0, has_bias=True)]
        classifier = nn.SequentialCell(classifier)

        # classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def construct(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class resnet50_add_layer5(nn.Cell):

    def __init__(self,
                 block=ResidualBlock,
                 layer_nums=None,
                 in_channels=None,
                 out_channels=None,
                 strides=None,
                 num_classes=10,
                 use_se=False,
                 res_base=False,
                 num_bottleneck=256):
        super(resnet50_add_layer5, self).__init__()
        if strides is None:
            strides = [1, 2, 2, 2, 1]
        if out_channels is None:
            out_channels = [256, 512, 1024, 2048]
        if in_channels is None:
            in_channels = [64, 256, 512, 1024]
        if layer_nums is None:
            layer_nums = [3, 4, 6, 3]
        self.part = 6
        self.feats = 256
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = P.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer3_1 = self._make_layer(block,
                                         layer_nums[2],
                                         in_channel=in_channels[2],
                                         out_channel=out_channels[2],
                                         stride=strides[2],
                                         use_se=self.use_se,
                                         se_block=self.se_block)
        self.layer3_2 = self._make_layer(block,
                                         layer_nums[2],
                                         in_channel=in_channels[2],
                                         out_channel=out_channels[2],
                                         stride=strides[2],
                                         use_se=self.use_se,
                                         se_block=self.se_block)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer5 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[4],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer6 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[4],
                                       use_se=self.use_se,
                                       se_block=self.se_block)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):

        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


import sys
import moxing as mox


class MGN(nn.Cell):

    def __init__(self, num_classes=10, num_bottleneck=256):
        super(MGN, self).__init__()
        feats = 2048
        self.norm = nn.BatchNorm1d(num_features=2048)
        resnet = resnet50_add_layer5(num_classes=1000)
        self.cat = ops.Concat(axis=1)
        # pretrain_chechpoint = 'C:/Users/ThinkPad/PycharmProjects/pythonProject/code-obs/MGN_mindspore_new.ckpt'
        # preckp_path = os.path.join('/cache/user-job-dir/code-pcb', 'precheckpoint')
        preckp_path = os.path.join('/home/work/user-job-dir/code_pcb/', 'precheckpoint')

        if not os.path.exists(preckp_path):
            os.mkdir(preckp_path)
            print(preckp_path)
        mobspath = 'obs://lihd-bucket/pretrained_ckpt/'
        mox.file.copy_parallel(mobspath, preckp_path)
        pretrain_chechpoint = os.path.join(preckp_path, 'MGN_mindspore_new.ckpt')

        dic = mindspore.train.serialization.load_checkpoint(pretrain_chechpoint)
        mindspore.train.serialization.load_param_into_net(resnet, parameter_dict=dic)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer3_1 = resnet.layer3_1
        self.layer3_2 = resnet.layer3_2
        self.layer4 = resnet.layer4
        self.layer5 = resnet.layer5
        self.layer6 = resnet.layer6

        self.dropout = nn.Dropout(keep_prob=0.5)
        self.flatten = ops.Flatten()
        self.split_2 = ops.Split(2, 2)
        self.split_3 = ops.Split(2, 3)
        self.l3 = self.layer3[0]
        self.p1 = nn.SequentialCell(self.layer3[1:], self.layer4)
        self.p2 = nn.SequentialCell(self.layer3_1[1:], self.layer5)
        self.p3 = nn.SequentialCell(self.layer3_2[1:], self.layer6)

        self.maxpool_zg_p1 = nn.AvgPool2d(kernel_size=(12, 4), stride=1)
        self.maxpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
        self.maxpool_zg_p3 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
        self.maxpool_zp2 = nn.AvgPool2d(kernel_size=(12, 8), stride=12)
        self.maxpool_zp3 = nn.AvgPool2d(kernel_size=(8, 8), stride=8)

        weight_shape = (num_classes, 2048)
        weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))

        weight_shape_g = (num_classes, 2048)
        weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))

        self.classifier1 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier2 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier3 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)

        self.classifier4 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier5 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier6 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier7 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier8 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)

        # self.reduction = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction1 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction2 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction3 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction4 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction5 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction6 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction7 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        # self.reduction8 = nn.SequentialCell(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(feats), nn.ReLU())
        #
        # weight_shape = (num_classes, 256)
        # weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
        #
        # weight_shape_g = (num_classes, 2048)
        # weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))
        #
        # self.classifier1 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
        # self.classifier2 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
        # self.classifier3 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
        # self.classifier4 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
        # self.classifier5 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
        # self.classifier6 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
        # self.classifier7 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
        # self.classifier8 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3[0](x)
        x = self.l3(x)
        # p1 = self.layer3(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        train_output = []

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        # global_1_re = self.flatten(self.reduction1(zg_p1))
        # predict.append(global_1_re)
        #
        # global_2_re = self.flatten(self.reduction2(zg_p2))
        # predict.append(global_2_re)
        #
        # global_3_re = self.flatten(self.reduction3(zg_p3))
        # predict.append(global_3_re)

        zp2 = self.split_2(self.maxpool_zp2(p2))
        z0_p2 = zp2[0]
        z1_p2 = zp2[1]

        zp3 = self.split_3(self.maxpool_zp3(p3))
        z0_p3 = zp3[0]
        z1_p3 = zp3[1]
        z2_p3 = zp3[2]

        fg_p1 = self.flatten(zg_p1)

        fg_p2 = self.flatten(zg_p2)

        fg_p3 = self.flatten(zg_p3)

        f0_p2 = self.flatten(z0_p2)
        f1_p2 = self.flatten(z1_p2)
        f0_p3 = self.flatten(z0_p3)
        f1_p3 = self.flatten(z1_p3)
        f2_p3 = self.flatten(z2_p3)

        l_p1, f1 = self.classifier6(fg_p1)
        train_output.append(f1)
        l_p2, f2 = self.classifier7(fg_p2)
        train_output.append(f1)
        l_p3, f3 = self.classifier8(fg_p3)
        train_output.append(f1)

        train_output.append(l_p1)
        train_output.append(l_p2)
        train_output.append(l_p3)

        l0_p2, f4 = self.classifier1(f0_p2)
        train_output.append(l0_p2)
        l1_p2, f5 = self.classifier2(f1_p2)
        train_output.append(l1_p2)
        l0_p3, f6 = self.classifier3(f0_p3)
        train_output.append(l0_p3)
        l1_p3, f7 = self.classifier4(f1_p3)
        train_output.append(l1_p3)
        l2_p3, f8 = self.classifier5(f2_p3)
        train_output.append(l2_p3)

        predict = self.cat((fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3))
        return train_output


class MGN_test(nn.Cell):

    def __init__(self, num_classes=10, num_bottleneck=256):
        super(MGN_test, self).__init__()
        feats = 2048
        self.norm = nn.BatchNorm1d(num_features=2048)
        resnet = resnet50_add_layer5(num_classes=1000)
        self.cat = ops.Concat(axis=1)
        # pretrain_chechpoint = 'C:/Users/ThinkPad/PycharmProjects/pythonProject/code-obs/MGN_mindspore_new.ckpt'
        # preckp_path = os.path.join('/cache/user-job-dir/code-pcb', 'precheckpoint')
        preckp_path = os.path.join('/home/work/user-job-dir/code_pcb', 'precheckpoint')

        if not os.path.exists(preckp_path):
            os.mkdir(preckp_path)
            print(preckp_path)
        mobspath = 'obs://lihd-bucket/pretrained_ckpt/'
        mox.file.copy_parallel(mobspath, preckp_path)
        pretrain_chechpoint = os.path.join(preckp_path, 'MGN_mindspore_new.ckpt')

        dic = mindspore.train.serialization.load_checkpoint(pretrain_chechpoint)
        mindspore.train.serialization.load_param_into_net(resnet, parameter_dict=dic)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer3_1 = resnet.layer3_1
        self.layer3_2 = resnet.layer3_2
        self.layer4 = resnet.layer4
        self.layer5 = resnet.layer5
        self.layer6 = resnet.layer6

        self.dropout = nn.Dropout(keep_prob=0.5)
        self.flatten = ops.Flatten()
        self.split_2 = ops.Split(2, 2)
        self.split_3 = ops.Split(2, 3)
        self.l3 = self.layer3[0]
        self.p1 = nn.SequentialCell(self.layer3[1:], self.layer4)
        self.p2 = nn.SequentialCell(self.layer3_1[1:], self.layer5)
        self.p3 = nn.SequentialCell(self.layer3_2[1:], self.layer6)

        self.maxpool_zg_p1 = nn.AvgPool2d(kernel_size=(12, 4), stride=1)
        self.maxpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
        self.maxpool_zg_p3 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
        self.maxpool_zp2 = nn.AvgPool2d(kernel_size=(12, 8), stride=12)
        self.maxpool_zp3 = nn.AvgPool2d(kernel_size=(8, 8), stride=8)

        weight_shape = (num_classes, 2048)
        weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))

        weight_shape_g = (num_classes, 2048)
        weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))

        self.classifier1 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier2 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier3 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)

        self.classifier4 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier5 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier6 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier7 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)
        self.classifier8 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
                                      num_bottleneck=num_bottleneck, return_f=True)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3[0](x)
        x = self.l3(x)
        # p1 = self.layer3(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        train_output = []

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        # global_1_re = self.flatten(self.reduction1(zg_p1))
        # predict.append(global_1_re)
        #
        # global_2_re = self.flatten(self.reduction2(zg_p2))
        # predict.append(global_2_re)
        #
        # global_3_re = self.flatten(self.reduction3(zg_p3))
        # predict.append(global_3_re)

        zp2 = self.split_2(self.maxpool_zp2(p2))
        z0_p2 = zp2[0]
        z1_p2 = zp2[1]

        zp3 = self.split_3(self.maxpool_zp3(p3))
        z0_p3 = zp3[0]
        z1_p3 = zp3[1]
        z2_p3 = zp3[2]

        fg_p1 = self.flatten(zg_p1)
        fg_p2 = self.flatten(zg_p2)
        fg_p3 = self.flatten(zg_p3)
        f0_p2 = self.flatten(z0_p2)
        f1_p2 = self.flatten(z1_p2)
        f0_p3 = self.flatten(z0_p3)
        f1_p3 = self.flatten(z1_p3)
        f2_p3 = self.flatten(z2_p3)

        l_p1, f1 = self.classifier6(fg_p1)
        train_output.append(f1)
        l_p2, f2 = self.classifier7(fg_p2)
        train_output.append(f1)
        l_p3, f3 = self.classifier8(fg_p3)
        train_output.append(f1)

        train_output.append(l_p1)
        train_output.append(l_p2)
        train_output.append(l_p3)

        l0_p2, f4 = self.classifier1(f0_p2)
        train_output.append(l0_p2)
        l1_p2, f5 = self.classifier2(f1_p2)
        train_output.append(l1_p2)
        l0_p3, f6 = self.classifier3(f0_p3)
        train_output.append(l0_p3)
        l1_p3, f7 = self.classifier4(f1_p3)
        train_output.append(l1_p3)
        l2_p3, f8 = self.classifier5(f2_p3)
        train_output.append(l2_p3)

        predict = self.cat((f1, f2, f3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3))
        return predict


# class MGN(nn.Cell):
#
#     def __init__(self, num_classes=10, num_bottleneck=256):
#         super(MGN, self).__init__()
#         feats = 256
#         self.norm = nn.BatchNorm1d(num_features=2048)
#         resnet = resnet50_add_layer5(num_classes=1000)
#         self.cat = ops.Concat(axis=1)
#         # pretrain_chechpoint = 'C:/Users/ThinkPad/PycharmProjects/pythonProject/code-obs/MGN_mindspore_new.ckpt'
#         # preckp_path = os.path.join('/cache/user-job-dir/code-pcb', 'precheckpoint')
#         preckp_path = os.path.join('/home/work/user-job-dir/code_pcb/', 'precheckpoint')
#
#         if not os.path.exists(preckp_path):
#             os.mkdir(preckp_path)
#             print(preckp_path)
#         mobspath = 'obs://lihd-bucket/pretrained_ckpt/'
#         mox.file.copy_parallel(mobspath, preckp_path)
#         pretrain_chechpoint = os.path.join(preckp_path, 'MGN_mindspore_new.ckpt')
#
#         dic = mindspore.train.serialization.load_checkpoint(pretrain_chechpoint)
#         mindspore.train.serialization.load_param_into_net(resnet, parameter_dict=dic)
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer3_1 = resnet.layer3_1
#         self.layer3_2 = resnet.layer3_2
#         self.layer4 = resnet.layer4
#         self.layer5 = resnet.layer5
#         self.layer6 = resnet.layer6
#
#         self.dropout = nn.Dropout(keep_prob=0.5)
#         self.flatten = ops.Flatten()
#         self.split_2 = ops.Split(2, 2)
#         self.split_3 = ops.Split(2, 3)
#         self.l3 = self.layer3[0]
#         self.p1 = nn.SequentialCell(self.layer3[1:], self.layer4)
#         self.p2 = nn.SequentialCell(self.layer3_1[1:], self.layer5)
#         self.p3 = nn.SequentialCell(self.layer3_2[1:], self.layer6)
#
#         self.maxpool_zg_p1 = nn.AvgPool2d(kernel_size=(12, 4), stride=1)
#         self.maxpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
#         self.maxpool_zg_p3 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
#         self.maxpool_zp2 = nn.AvgPool2d(kernel_size=(12, 8), stride=12)
#         self.maxpool_zp3 = nn.AvgPool2d(kernel_size=(8, 8), stride=8)
#
#         weight_shape = (num_classes, 2048)
#         weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
#
#         weight_shape_g = (num_classes, 2048)
#         weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))
#
#         # self.classifier1 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier2 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier3 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         #
#         # self.classifier4 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier5 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier6 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier7 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier8 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#
#         self.reduction = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction1 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction2 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction3 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction4 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction5 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction6 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction7 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         # self.reduction8 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#
#         weight_shape = (num_classes, 256)
#         weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
#
#         weight_shape_g = (num_classes, 256)
#         weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))
#
#         self.classifier1 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier2 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier3 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier4 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier5 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier6 = nn.Dense(256, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#         self.classifier7 = nn.Dense(256, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#         self.classifier8 = nn.Dense(256, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#
#     def construct(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         # x = self.layer3[0](x)
#         x = self.l3(x)
#         # p1 = self.layer3(x)
#         p1 = self.p1(x)
#         p2 = self.p2(x)
#         p3 = self.p3(x)
#
#         train_output = []
#
#         zg_p1 = self.maxpool_zg_p1(p1)
#         zg_p2 = self.maxpool_zg_p2(p2)
#         zg_p3 = self.maxpool_zg_p3(p3)
#
#         zp2 = self.split_2(self.maxpool_zp2(p2))
#         z0_p2 = zp2[0]
#         z1_p2 = zp2[1]
#
#         zp3 = self.split_3(self.maxpool_zp3(p3))
#         z0_p3 = zp3[0]
#         z1_p3 = zp3[1]
#         z2_p3 = zp3[2]
#
#         fg_p1 = self.flatten(self.reduction(zg_p1))
#         train_output.append(fg_p1)
#         fg_p2 = self.flatten(self.reduction(zg_p2))
#         train_output.append(fg_p1)
#         fg_p3 = self.flatten(self.reduction(zg_p3))
#         train_output.append(fg_p1)
#         f0_p2 = self.flatten(self.reduction(z0_p2))
#         f1_p2 = self.flatten(self.reduction(z1_p2))
#         f0_p3 = self.flatten(self.reduction(z0_p3))
#         f1_p3 = self.flatten(self.reduction(z1_p3))
#         f2_p3 = self.flatten(self.reduction(z2_p3))
#
#         l_p1 = self.classifier6(fg_p1)
#         train_output.append(l_p1)
#         l_p2 = self.classifier7(fg_p2)
#         train_output.append(l_p2)
#         l_p3 = self.classifier8(fg_p3)
#         train_output.append(l_p3)
#
#         l0_p2 = self.classifier1(f0_p2)
#         train_output.append(l0_p2)
#         l1_p2 = self.classifier2(f1_p2)
#         train_output.append(l1_p2)
#         l0_p3 = self.classifier3(f0_p3)
#         train_output.append(l0_p3)
#         l1_p3 = self.classifier4(f1_p3)
#         train_output.append(l1_p3)
#         l2_p3 = self.classifier5(f2_p3)
#         train_output.append(l2_p3)
#
#         predict = self.cat((fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3))
#         return train_output
#
# class MGN_test(nn.Cell):
#
#     def __init__(self, num_classes=10, num_bottleneck=256):
#         super(MGN_test, self).__init__()
#         feats = 256
#         self.norm = nn.BatchNorm1d(num_features=2048)
#         resnet = resnet50_add_layer5(num_classes=1000)
#         self.cat = ops.Concat(axis=1)
#         # pretrain_chechpoint = 'C:/Users/ThinkPad/PycharmProjects/pythonProject/code-obs/MGN_mindspore_new.ckpt'
#         # preckp_path = os.path.join('/cache/user-job-dir/code-pcb', 'precheckpoint')
#         preckp_path = os.path.join(sys.path[0], 'precheckpoint')
#
#         if not os.path.exists(preckp_path):
#             os.mkdir(preckp_path)
#             print(preckp_path)
#         mobspath = 'obs://lihd-bucket/pretrained_ckpt/'
#         mox.file.copy_parallel(mobspath, preckp_path)
#         pretrain_chechpoint = os.path.join(preckp_path, 'MGN_mindspore_new.ckpt')
#
#         dic = mindspore.train.serialization.load_checkpoint(pretrain_chechpoint)
#         mindspore.train.serialization.load_param_into_net(resnet, parameter_dict=dic)
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer3_1 = resnet.layer3_1
#         self.layer3_2 = resnet.layer3_2
#         self.layer4 = resnet.layer4
#         self.layer5 = resnet.layer5
#         self.layer6 = resnet.layer6
#
#         self.dropout = nn.Dropout(keep_prob=0.5)
#         self.flatten = ops.Flatten()
#         self.split_2 = ops.Split(2, 2)
#         self.split_3 = ops.Split(2, 3)
#         self.l3 = self.layer3[0]
#         self.p1 = nn.SequentialCell(self.layer3[1:], self.layer4)
#         self.p2 = nn.SequentialCell(self.layer3_1[1:], self.layer5)
#         self.p3 = nn.SequentialCell(self.layer3_2[1:], self.layer6)
#
#         self.maxpool_zg_p1 = nn.AvgPool2d(kernel_size=(12, 4), stride=1)
#         self.maxpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
#         self.maxpool_zg_p3 = nn.AvgPool2d(kernel_size=(24, 8), stride=1)
#         self.maxpool_zp2 = nn.AvgPool2d(kernel_size=(12, 8), stride=12)
#         self.maxpool_zp3 = nn.AvgPool2d(kernel_size=(8, 8), stride=8)
#
#         weight_shape = (num_classes, 2048)
#         weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
#
#         weight_shape_g = (num_classes, 2048)
#         weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))
#
#         # self.classifier1 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier2 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier3 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         #
#         # self.classifier4 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier5 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier6 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier7 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#         # self.classifier8 = ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True,
#         #                               num_bottleneck=num_bottleneck)
#
#         self.reduction = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction1 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction2 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction3 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction4 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction5 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction6 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction7 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#         self.reduction8 = nn.SequentialCell(nn.Conv2d(2048, 256, 1, weight_init='he_uniform'), nn.BatchNorm2d(feats), nn.ReLU())
#
#         weight_shape = (num_classes, 256)
#         weight = Tensor(kaiming_normal(weight_shape, a=0, mode='fan_out'))
#
#         weight_shape_g = (num_classes, 2048)
#         weight_g = Tensor(kaiming_normal(weight_shape_g, a=0, mode='fan_out'))
#
#         self.classifier1 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier2 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier3 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier4 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier5 = nn.Dense(256, num_classes, weight_init=weight, bias_init=0, has_bias=True)
#         self.classifier6 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#         self.classifier7 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#         self.classifier8 = nn.Dense(2048, num_classes, weight_init=weight_g, bias_init=0, has_bias=True)
#
#     def construct(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         # x = self.layer3[0](x)
#         x = self.l3(x)
#         # p1 = self.layer3(x)
#         p1 = self.p1(x)
#         p2 = self.p2(x)
#         p3 = self.p3(x)
#
#         train_output = []
#
#         zg_p1 = self.maxpool_zg_p1(p1)
#         zg_p2 = self.maxpool_zg_p2(p2)
#         zg_p3 = self.maxpool_zg_p3(p3)
#
#         zp2 = self.split_2(self.maxpool_zp2(p2))
#         z0_p2 = zp2[0]
#         z1_p2 = zp2[1]
#
#         zp3 = self.split_3(self.maxpool_zp3(p3))
#         z0_p3 = zp3[0]
#         z1_p3 = zp3[1]
#         z2_p3 = zp3[2]
#
#         fg_p1 = self.flatten(zg_p1)
#         fg_p2 = self.flatten(zg_p2)
#         fg_p3 = self.flatten(zg_p3)
#         f0_p2 = self.flatten(self.reduction4(z0_p2))
#         f1_p2 = self.flatten(self.reduction5(z1_p2))
#         f0_p3 = self.flatten(self.reduction6(z0_p3))
#         f1_p3 = self.flatten(self.reduction7(z1_p3))
#         f2_p3 = self.flatten(self.reduction8(z2_p3))
#
#         l_p1 = self.classifier6(fg_p1)
#         train_output.append(l_p1)
#         l_p2 = self.classifier7(fg_p2)
#         train_output.append(l_p2)
#         l_p3 = self.classifier8(fg_p3)
#         train_output.append(l_p3)
#
#         l0_p2 = self.classifier1(f0_p2)
#         train_output.append(l0_p2)
#         l1_p2 = self.classifier2(f1_p2)
#         train_output.append(l1_p2)
#         l0_p3 = self.classifier3(f0_p3)
#         train_output.append(l0_p3)
#         l1_p3 = self.classifier4(f1_p3)
#         train_output.append(l1_p3)
#         l2_p3 = self.classifier5(f2_p3)
#         train_output.append(l2_p3)
#
#         predict = self.cat((fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3))
#         return predict

if __name__ == '__main__':
    # model=PCB(10)

    model = MGN(num_classes=1000)
    #     print(model)
    input = Tensor(np.ones([8, 3, 384, 128]).astype("float32"))
    out = model(input)
    # loss = MGN_Loss(batch=8, thres=Tensor(np.zeros((8, 8))))
    # target = Tensor(np.array([1, 1, 3, 4, 3, 5, 2, 2]))
    # out = loss.construct(out, target)

    print(out)
