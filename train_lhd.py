# 要写读取数据
# 模型
# 损失函数
# 用model把模型和损失函数结合起来
# 训练时要有监控
# 设置训练的参数
# 多卡可以自动合并保存，不用考虑切片啥的
"""train resnet."""
import mindspore.ops as ops
import os
import argparse
import ast
from mindspore import context
from mindspore import Tensor
from test2 import test_v1
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from test_byrerank import test_byre
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from test_byrerank import test_byre
from mindspore.train.callback._lr_scheduler_callback import LearningRateScheduler
from mindspore.train.callback import Callback, SummaryCollector
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

parser = argparse.ArgumentParser(description='reidentification')

parser.add_argument('--dataset', type=str, default=None, help='Dataset, either cifar10 or imagenet2012')

parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')  # 显卡个数
parser.add_argument('--batch_size', type=int, default=32, help='dataset batch size.')
parser.add_argument('--epoch_size', type=int, default=200, help='train epoch.')
parser.add_argument('--dataset_path_train', type=str, default='./dataset/market/train', help='Dataset path')  # 读取数据的位置
parser.add_argument('--dataset_path_val', type=str, default='./dataset/market/val', help='Dataset path')  # 读取数据的位置
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--lr_rate', default=0.1, type=float, help='bachbone learing rate mult this number')
parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='dyle decay rate')
parser.add_argument('--lr_decay_epoch', default=40, type=int, help='dyle decay epoch')

parser.add_argument('--checkpoint_save_path', type=str, default='./chechpoint', help='Dataset path')

# 为了跑起来才写的
parser.add_argument('--train_url', type=str, default='./chechpoint', help='Dataset path')
parser.add_argument('--data_url', type=str, default='./chechpoint', help='Dataset path')

parser.add_argument('--save_checkpoint_epochs', type=int, default=3, help='dataset batch size.')

parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")  # 什么平台
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                    help="Filter head weight parameters, default is False.")
args_opt = parser.parse_args()

set_seed(1)
from model_pcb import PCB_with_resnet as pcbnet
from model_pcb import PCB_with_resnet_V1, PCB_with_resnet_V384
from dataset import create_dataset_384
from pcb_loss import PCBLoss, PCBLoss_v2, PCBLoss_lhd
from abc import ABCMeta, abstractmethod
from loss import MGN_Loss
import numpy as np
from mindspore.common.tensor import Tensor
from MGN_with_PCB import MGN

_eval_types = {'classification', 'multilabel'}


class Metric(metaclass=ABCMeta):
    """
    Base class of metric.


    Note:
        For examples of subclasses, please refer to the definition of class `MAE`, 'Recall' etc.
    """

    def __init__(self):
        pass

    def _convert_data(self, data):
        """
        Convert data type to numpy array.

        Args:
            data (Object): Input data.

        Returns:
            Ndarray, data with `np.ndarray` type.
        """
        if isinstance(data, Tensor):
            data = data.asnumpy()
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError('Input data type must be tensor, list or numpy.ndarray')
        return data

    def _check_onehot_data(self, data):
        """
        Whether input data are one-hot encoding.

        Args:
            data (numpy.array): Input data.

        Returns:
            bool, return true, if input data are one-hot encoding.
        """
        if data.ndim > 1 and np.equal(data ** 2, data).all():
            shp = (data.shape[0],) + data.shape[2:]
            if np.equal(np.ones(shp), data.sum(axis=1)).all():
                return True
        return False

    def _binary_clf_curve(self, preds, target, sample_weights=None, pos_label=1):
        """Calculate True Positives and False Positives per binary classification threshold."""
        if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
            sample_weights = np.array(sample_weights)

        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = np.argsort(-preds)

        preds = preds[desc_score_indices]
        target = target[desc_score_indices]

        if sample_weights is not None:
            weight = sample_weights[desc_score_indices]
        else:
            weight = 1.

        distinct_value_indices = np.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = np.pad(distinct_value_indices, (0, 1), constant_values=target.shape[0] - 1)
        target = np.array(target == pos_label).astype(np.int64)
        tps = np.cumsum(target * weight, axis=0)[threshold_idxs]

        if sample_weights is not None:
            fps = np.cumsum((1 - target) * weight, axis=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps

        return fps, tps, preds[threshold_idxs]

    def __call__(self, *inputs):
        """
        Evaluate input data once.

        Args:
            inputs (tuple): The first item is predict array, the second item is target array.

        Returns:
            Float, compute result.
        """
        self.clear()
        self.update(*inputs)
        return self.eval()

    @abstractmethod
    def clear(self):
        """
        An interface describes the behavior of clearing the internal evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError('Must define clear function to use this base class')

    @abstractmethod
    def eval(self):
        """
        An interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError('Must define eval function to use this base class')

    @abstractmethod
    def update(self, *inputs):
        """
        An interface describes the behavior of updating the internal evaluation result.

        Note:
            All subclasses must override this interface.

        Args:
            inputs: A variable-length input argument list.
        """
        raise NotImplementedError('Must define update function to use this base class')


class EvaluationBase(Metric):
    """
    Base class of evaluation.

    Note:
        Please refer to the definition of class `Accuracy`.

    Args:
        eval_type (str): Type of evaluation must be in {'classification', 'multilabel'}.

    Raises:
        TypeError: If the input type is not classification or multilabel.
    """

    def __init__(self, eval_type):
        super(EvaluationBase, self).__init__()
        if eval_type not in _eval_types:
            raise TypeError('Type must be in {}, but got {}'.format(_eval_types, eval_type))
        self._type = eval_type

    def _check_shape(self, y_pred, y):
        """
        Checks the shapes of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type == 'classification':
            if y_pred.ndim != y.ndim + 1:
                raise ValueError('Classification case, dims of y_pred equal dims of y add 1, '
                                 'but got y_pred: {} dims and y: {} dims'.format(y_pred.ndim, y.ndim))
            if y.shape != (y_pred.shape[0],) + y_pred.shape[2:]:
                raise ValueError('Classification case, y_pred shape and y shape can not match. '
                                 'got y_pred shape is {} and y shape is {}'.format(y_pred.shape, y.shape))
        else:
            if y_pred.ndim != y.ndim:
                raise ValueError('{} case, dims of y_pred need equal with dims of y, but got y_pred: {} '
                                 'dims and y: {} dims.'.format(self._type, y_pred.ndim, y.ndim))
            if y_pred.shape != y.shape:
                raise ValueError('{} case, y_pred shape need equal with y shape, but got y_pred: {} and y: {}'.
                                 format(self._type, y_pred.shape, y.shape))

    def _check_value(self, y_pred, y):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type != 'classification' and not (np.equal(y_pred ** 2, y_pred).all() and np.equal(y ** 2, y).all()):
            raise ValueError('For multilabel case, input value must be 1 or 0.')

    def clear(self):
        """
        A interface describes the behavior of clearing the internal evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError

    def update(self, *inputs):
        """
        A interface describes the behavior of updating the internal evaluation result.

        Note:
            All subclasses must override this interface.

        Args:
            inputs: The first item is predicted array and the second item is target array.
        """
        raise NotImplementedError

    def eval(self):
        """
        A interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError


import numpy as np


# from mindspore.nn.metrics.metric import EvaluationBase

# 自己写的一个accuracy，用于看测试集的分类准确率

class Accuracy_pcb(nn.metrics.Metric):
    r"""
    Calculates the accuracy for classification and multilabel data.

    The accuracy class creates two local variables, the correct number and the total number that are used to compute the
    frequency with which predictions matches labels. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides the correct number by the total number.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for
            classification (single-label), and multilabel (multilabel classification).
            Default: 'classification'.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> y = Tensor(np.array([1, 0, 1]), mindspore.float32)
        >>> metric = nn.Accuracy('classification')
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> accuracy = metric.eval()
        >>> print(accuracy)
        0.6666666666666666
    """

    def __init__(self, eval_type='classification'):
        super(Accuracy_pcb, self).__init__()
        if eval_type not in _eval_types:
            raise TypeError('Type must be in {}, but got {}'.format(_eval_types, eval_type))
        self._type = eval_type
        self.softmax = ops.Softmax()
        self.clear()
        self.scalar_summary = ops.ScalarSummary()

    def _check_shape(self, y_pred, y):
        """
        Checks the shapes of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type == 'classification':
            if y_pred.ndim != y.ndim + 1:
                raise ValueError('Classification case, dims of y_pred equal dims of y add 1, '
                                 'but got y_pred: {} dims and y: {} dims'.format(y_pred.ndim, y.ndim))
            if y.shape != (y_pred.shape[0],) + y_pred.shape[2:]:
                raise ValueError('Classification case, y_pred shape and y shape can not match. '
                                 'got y_pred shape is {} and y shape is {}'.format(y_pred.shape, y.shape))
        else:
            if y_pred.ndim != y.ndim:
                raise ValueError('{} case, dims of y_pred need equal with dims of y, but got y_pred: {} '
                                 'dims and y: {} dims.'.format(self._type, y_pred.ndim, y.ndim))
            if y_pred.shape != y.shape:
                raise ValueError('{} case, y_pred shape need equal with y shape, but got y_pred: {} and y: {}'.
                                 format(self._type, y_pred.shape, y.shape))

    def _check_value(self, y_pred, y):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type != 'classification' and not (np.equal(y_pred ** 2, y_pred).all() and np.equal(y ** 2, y).all()):
            raise ValueError('For multilabel case, input value must be 1 or 0.')

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, a list or an array.
                For the 'classification' evaluation type, `y_pred` is in most cases (not strictly) a list
                of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. Shape of `y` can be :math:`(N, C)` with values 0 and 1 if one-hot
                encoding is used or the shape is :math:`(N,)` with integer values if index of category is used.
                For 'multilabel' evaluation type, `y_pred` and `y` can only be one-hot encoding with
                values 0 or 1. Indices with 1 indicate the positive category. The shape of `y_pred` and `y`
                are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Accuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_temp = inputs[0]
        y_t = self.softmax(y_temp[3])
        for i in range(4, 11):
            y_t += self.softmax(y_temp[i])

        y = self._convert_data(inputs[1])
        y_pred = self._convert_data(y_t)
        # print(y)
        # print(y_pred)
        # print(y)
        # print(y_pred)
        if self._type == 'classification' and y_pred.ndim == y.ndim and self._check_onehot_data(y):
            y = y.argmax(axis=1)
            print(y)
        self._check_shape(y_pred, y)
        self._check_value(y_pred, y)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._class_num, y_pred.shape[1]))

        if self._type == 'classification':
            indices = y_pred.argmax(axis=1)
            # print(indices)
            result = (np.equal(indices, y) * 1).reshape(-1)
            # print(result)
        elif self._type == 'multilabel':
            dimension_index = y_pred.ndim - 1
            y_pred = y_pred.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            y = y.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            result = np.equal(y_pred, y).all(axis=1) * 1

        self._correct_num += result.sum()
        self._total_num += result.shape[0]

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError('Accuary can not be calculated, because the number of samples is 0.')
        acc = self._correct_num / self._total_num
        # self.scalar_summary("acc", acc)
        # return self._correct_num / self._total_num
        return acc


from mindspore.train.callback import Callback


# custom callback function
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, datanum):
        self.model = model
        self.eval_dataset = eval_dataset

        self.satanum = datanum

    # 每次都顺便复制一下保存的断电
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        cur_epoch = cb_params.cur_epoch_num

        cur_step = (cur_epoch - 1) * self.satanum + cb_params.cur_step_num
        if cur_step % 5 == 0:
            res = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            acc = res['accuracy']
            loss = res['loss']
            print('val Loss: {:.4f} Acc: {:.4f}'.format(
                loss, acc))
            # print('train Loss: {:.4f} '.format(
            #     cb_params.net_outputs))


class Evaluate_v1(Callback):
    def __init__(self):
        self.best_t1 = 0
        self.best_epoch = 0

    # 每次都顺便复制一下保存的断电
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch % 5 == 2:
            if cur_epoch > 5:

                cmc, map = test_byre(cur_epoch)
                if cmc[0] > self.best_t1:
                    self.best_t1 = cmc[0]
                    self.best_epoch = cur_epoch - 2
        print('now best epoch is %d ,top1 is %f' % (self.best_epoch, self.best_t1))


class CopyChevkpoint(Callback):
    def __init__(self, modelaret_save_path=None, obs_save_path=None, savefrea=None):
        self.modelart = modelaret_save_path
        self.obs = obs_save_path
        self.savetime = savefrea

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch % self.savetime == 0:
            mox.file.copy_parallel(self.modelart, self.obs)


class Copysummery(Callback):
    def __init__(self, summery_save_path=None, obs_save_path=None, savefrea=None):
        self.summery = summery_save_path
        self.obs = obs_save_path
        self.savetime = savefrea

    # 每次都顺便复制一下保存的断电
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        # if cur_epoch % self.savetime == 0:
        mox.file.copy_parallel(self.summery, self.obs)


import os
import stat
import time

import threading
import mindspore.context as context
from mindspore import log as logger
from mindspore import nn
from mindspore._checkparam import Validator
from mindspore.train._utils import _make_directory
from mindspore.train.serialization import save_checkpoint, _save_graph
from mindspore.parallel._ps_context import _is_role_pserver, _get_ps_mode_rank

import moxing as mox


def dowload_data_set(obspath='obs://lihd-bucket/ew_Mar/', datasetname='market'):
    import sys
    import moxing as mox
    localpath = os.path.join('/home/work/user-job-dir/code_pcb/', 'dataset')
    print(localpath)
    if not os.path.exists(localpath):
        os.mkdir(localpath)
    dpath = os.path.join(localpath, datasetname)
    print(dpath)
    mobspath = obspath
    if not os.path.exists(dpath):
        os.mkdir(dpath)
    mox.file.copy_parallel(mobspath, dpath)


def dowload_data_set2(obspath='obs://lihd-bucket/dataset/Market-1501-v15.09.15/',
                      dataset_name='market'):  # 'ew_Mar' or 'data_demo'
    print(obspath)
    localpath = os.path.join(sys.path[0], 'dataset')
    # localpath = os.path.join('/cache/user-job-dir/code-obs', 'dataset')
    print(localpath)
    if not os.path.exists(localpath):
        os.mkdir(localpath)
    dpath = os.path.join(localpath, dataset_name)
    print(dpath)
    mobspath = obspath
    if not os.path.exists(dpath):
        os.mkdir(dpath)
    mox.file.copy_parallel(mobspath, dpath)


import sys
import moxing as mox
import mindspore


def learning_rate_function(lr, cur_step_num, epoch_down=20, steps=190):
    if cur_step_num % (epoch_down * steps) == 0 or cur_step_num % (60 * steps) == 0:
        lr = lr * 0.1
    return lr


if __name__ == '__main__':
    dowload_data_set()
    target = args_opt.device_target
    # if target == "CPU":
    #     args_opt.run_distribute = False
    dataset_path = os.path.join('/home/work/user-job-dir/code_pcb', 'dataset')
    dataset_path = os.path.join(dataset_path, 'market')
    dataset_path_train = os.path.join(dataset_path, 'train')
    dataset_path_val = os.path.join(dataset_path, 'val')
    # dataset_path_train = os.path.join(dataset_path, 'bounding_box_train')
    # dataset_path_val = os.path.join(dataset_path, 'query')
    # ckpt_save_dir = os.path.join(sys.path[0], 'checkpoint')
    ckpt_save_dir = os.path.join('/home/work/user-job-dir/code_pcb', 'checkpoint')
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
        print(ckpt_save_dir)

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    # if args_opt.parameter_server:
    #     context.set_ps_context(enable_ps=True)
    device_id = int(os.getenv('DEVICE_ID'))
    # context.set_context(device_id=3, enable_auto_mixed_precision=True)
    context.set_context(enable_auto_mixed_precision=True)
    print(context.get_context("device_id"))
    # context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
    #                                           gradients_mean=True)
    # set_algo_parameters(elementwise_op_strategy_follow=True)
    # context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])

    # init()

    # create dataset
    dataset_train = create_dataset_384(dataset_path=dataset_path_train, do_train=True, repeat_num=1,
                                       batxhsize=args_opt.batch_size, platform=target)
    step_size = dataset_train.get_dataset_size()
    dataset_val = create_dataset_384(dataset_path=dataset_path_val, do_train=False, repeat_num=1,
                                     batxhsize=1, platform=target)
    # define net
    # preckp_path = os.path.join(sys.path[0], 'precheckpoint')
    preckp_path = os.path.join('/home/work/user-job-dir/code_pcb/', 'precheckpoint')
    if not os.path.exists(preckp_path):
        os.mkdir(preckp_path)
        print(preckp_path)
    # import sys
    # import moxing as mox
    # mobspath = 'obs://wangl97/pretrained_ckpt/'
    # mox.file.copy_parallel(mobspath, preckp_path)
    # pretrain_chechpoint = os.path.join(preckp_path, 'resnet50_frompytorch.ckpt')
    # pretrain_classifier = os.path.join(preckp_path, 'classifier.ckpt')

    classes = dataset_train.num_classes()
    print(classes)
    # net = pcbnet(num_classes=classes)
    # net = PCB_with_resnet_V384(num_classes=classes, num_bottleneck=1024)  # 数据集有多少泪？？？
    net = MGN(num_classes=classes)
    # dic = mindspore.train.serialization.load_checkpoint(pretrain_chechpoint)
    # dic_classifier = mindspore.train.serialization.load_checkpoint(pretrain_classifier)
    # mindspore.train.serialization.load_param_into_net(net,parameter_dict=dic)
    # mindspore.train.serialization.load_param_into_net(net, parameter_dict=dic_classifier)

    # print(net.trainable_params())
    # weight_decay=5e-4, momentum=0.9, nesterov=True
    classifier_param = []
    resnet_back_bone_param = []
    for param in net.trainable_params():
        if 'classifier' not in param.name and 'add_block' not in param.name:
            resnet_back_bone_param.append(param)
        else:
            classifier_param.append(param)

    gropu_param = [{'params': resnet_back_bone_param, 'lr': args_opt.lr * args_opt.lr_rate},
                   {'params': classifier_param, 'lr': args_opt.lr},
                   {'order_params': net.trainable_params()}
                   ]

    # backbonel_decay_lr = mindspore.nn.ExponentialDecayLR(args_opt.lr * args_opt.lr_rate, args_opt.lr_decay_rate,
    #                                                      args_opt.lr_decay_epoch * step_size)
    # classifier_decay_lr = mindspore.nn.ExponentialDecayLR(args_opt.lr, args_opt.lr_decay_rate,
    #                                                       args_opt.lr_decay_epoch * step_size)
    # gropu_param = [{'params': resnet_back_bone_param, 'lr': backbonel_decay_lr},
    #                {'params': classifier_param, 'lr': classifier_decay_lr},
    #                {'order_params': net.trainable_params()}
    #                ]
    # print(step_size)
    # backbonel_decay_lr = mindspore.nn.exponential_decay_lr(args_opt.lr * args_opt.lr_rate,
    #                                                        args_opt.lr_decay_rate,
    #                                                        args_opt.epoch_size * step_size,
    #                                                        step_size,
    #                                                        args_opt.lr_decay_epoch,
    #                                                        is_stair=True)
    # classifier_decay_lr = mindspore.nn.exponential_decay_lr(args_opt.lr,
    #                                                         args_opt.lr_decay_rate,
    #                                                         args_opt.epoch_size * step_size,
    #                                                         step_size,
    #                                                         args_opt.lr_decay_epoch,
    #                                                         is_stair=True)
    # # print(backbonel_decay_lr)
    # # print(classifier_decay_lr)
    # gropu_param = [{'params': resnet_back_bone_param, 'lr': backbonel_decay_lr},
    #                {'params': classifier_param, 'lr': classifier_decay_lr},
    #                {'order_params': net.trainable_params()}
    #                ]
    '''
    resnet层的参数取出来
    classifier层的参数取出来
    '''
    # optim=nn.SGD(net.trainable_params(),learning_rate=args_opt.lr,weight_decay=5e-4, momentum=0.9,nesterov=True)
    optim = nn.SGD(gropu_param, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # optim=nn.Momentum(gropu_param,learning_rate=args_opt.lr, momentum=0.9, weight_decay=5e-4,use_nesterov=True)
    # define loss, model
    '''
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
            '''
    loss = MGN_Loss(batch=args_opt.batch_size)
    # 这tm怎么自己写准确率啊
    metrics = {
        'accuracy': Accuracy_pcb(),
        'loss': nn.Loss()

    }
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=16384)
    # # model = Model(net, loss_fn=loss, optimizer=optim,  metrics=metrics,
    #               amp_level="O3", keep_batchnorm_fp32=False)#為什麼要把網絡整成这样
    # 写一下加载模型的程序
    model = Model(net, loss_scale_manager=loss_scale_manager, loss_fn=loss, optimizer=optim, metrics=metrics,
                  amp_level="O0", keep_batchnorm_fp32=False)  # o3会怎么样

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    # ev = Evaluate_v1()
    ###
    ###
    ###
    stepl = StepLossAccInfo(model, dataset_val, step_size)  # 注意这里写一下
    save = CopyChevkpoint(modelaret_save_path=ckpt_save_dir, obs_save_path=args_opt.train_url,
                          savefrea=args_opt.save_checkpoint_epochs)

    cb = [time_cb, loss_cb, stepl]

    config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_epochs * step_size,
                                 )
    ckpt_cb = ModelCheckpoint(prefix="pcbnet", directory=ckpt_save_dir, config=config_ck)
    # 可以全训练晚了就复制一波
    cb += [ckpt_cb]
    cb += [save]
    # cb += [ev]
    LRS = LearningRateScheduler(learning_rate_function)
    summary_dir = os.path.join('/home/work/user-job-dir/code_pcb/', 'summary_dir')
    print(summary_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    summary_collector = SummaryCollector(summary_dir=summary_dir)
    save_summery = Copysummery(summery_save_path=summary_dir, obs_save_path=args_opt.train_url,
                               savefrea=args_opt.save_checkpoint_epochs)
    cb += [summary_collector]
    cb += [save_summery]
    # # train model
    # if args_opt.net == "se-resnet50":
    #     config.epoch_size = config.train_epoch_size
    # dataset_sink_mode = (not args_opt.parameter_server) and target != "CPU"
    # model.train(args_opt.epoch_size , dataset_train, callbacks=cb,
    #             sink_size=dataset_train.get_dataset_size(), dataset_sink_mode=True)
    model.train(args_opt.epoch_size, dataset_train, callbacks=cb, sink_size=-1, dataset_sink_mode=True)
    # try:
    # model.train(args_opt.epoch_size, dataset_train, callbacks=cb,  dataset_sink_mode=False)
    # except:
    #     mox.file.copy_parallel('./task_error_dump', 'obs://wangl97/1/')
    # 想每个epoch后验证一下咋写
    # 可以自己写一个callback，每次epoch后都测试一下结果
    # 主要是自己要写一个accuracy，来测试
