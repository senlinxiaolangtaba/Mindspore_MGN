import mindspore.ops as ops
import mindspore
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import nn
from mindspore.ops import _selected_ops
import numpy as np
from mindspore.ops import Fill
from mindspore.ops import composite as C


# import mindspore.numpy as np


def expand(self, x):
    fill = Fill()
    temp = fill((mindspore.float32, (1, x.shape(0)), 1))
    output = [x1.expand_as(temp) for x1 in x]

    return output


class _Loss(nn.Cell):
    """
    Base class for other losses.
    """

    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = _selected_ops.ReduceMean()
        self.reduce_sum = P.ReduceSum()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, input, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = input.dtype
        input = self.cast(input, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        input = self.mul(weights, input)
        if self.reduce and self.average:
            input = self.reduce_mean(input, self.get_axis(input))
        if self.reduce and not self.average:
            input = self.reduce_sum(input, self.get_axis(input))
            # input = self.reduce_sum(input)
        input = self.cast(input, input_dtype)
        return input

    def construct(self, base, target):
        raise NotImplementedError


class TripletLoss(_Loss):

    def __init__(self, batch=32, perm=(1, 0), thres=None, min_value=None, max_value=None, margin=1.5, mutual_flag=False,
                 ):
        super(TripletLoss, self).__init__()
        if thres is None:
            thres = []
        if min_value == None:
            min_value = Tensor(0, mstype.float32)
        if max_value == None:
            max_value = Tensor(1, mstype.float32)
        self.min_value = min_value
        self.max_value = max_value
        self.batch = batch
        self.margin = margin
        self.maximum = ops.ArgMaxWithValue()
        self.minimum = ops.ArgMinWithValue()
        self.mutual = mutual_flag
        self.pow = ops.Pow()
        self.equal = ops.Equal()
        self.sqrt = ops.Sqrt()
        self.matrix_multiply = ops.MatMul()
        self.squeeze = ops.Squeeze(1)
        self.squeeze0 = ops.Squeeze(0)
        self.cumsum = ops.CumSum()
        self.transpose = ops.Transpose()
        self.perm = perm
        self.broadcastto = ops.BroadcastTo((self.batch, self.batch))
        self.mul = ops.Mul()
        self.not_equal = ops.NotEqual()
        self.thres = thres
        self.greater = ops.Greater()
        self.print = ops.Print()
        self.max = ops.Maximum()
        self.reducemesns = ops.ReduceMean()
        self.abs = ops.Abs()

    def construct(self, base, targets):
        # dist = self.broadcastto(self.squeeze(self.cumsum(self.pow(base, 2), 1)[:, -1:]))
        dist = self.broadcastto(self.reduce_sum(self.pow(base, 2), 1))
        dist = dist + self.transpose(dist, self.perm)
        xy = self.matrix_multiply(base, self.transpose(base, self.perm))
        dist = self.sqrt(self.abs(dist - 2 * (xy)) + 1e-4)
        mask = self.equal(self.broadcastto(targets), self.transpose(self.broadcastto(targets), self.perm))

        dist_ap = self.mul(dist, mask)
        temp = self.not_equal(dist_ap, dist)
        dist_an = self.mul(dist, temp) + mask * 10000000
        _, ap = self.maximum(dist_ap)
        _, an = self.minimum(dist_an)

        # loss = C.clip_by_value(self.max((ap - an + self.margin), 0), self.min_value, self.max_value)
        loss = self.max((ap - an + self.margin), 0)
        # loss = loss.sum()/len(loss)
        ret = self.reducemesns(loss, 0)
        # loss = self.squeeze0(ret[ret.shape[0] - 1:ret.shape[0]]) / self.batch

        return ret


class MGN_Loss(_Loss):

    def __init__(self, batch=64, thres=None, reduction='none'):
        super(MGN_Loss, self).__init__(reduction)

        if thres is None:
            thres = []
        self.scalar_summary = ops.ScalarSummary()
        self.cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.TripletLoss = TripletLoss(batch=batch, margin=1.5, thres=thres)
        self.squeeze = ops.Squeeze(0)
        # self.metric = nn.Accuracy('classification')/root/miniconda3

    def construct(self, base, target):
        TriLoss = self.TripletLoss(base[0], target)
        TriLoss += self.TripletLoss(base[1], target)
        TriLoss += self.TripletLoss(base[2], target)
        crossloss = self.cross_entropy_loss(base[3], target)
        crossloss += self.cross_entropy_loss(base[4], target)
        crossloss += self.cross_entropy_loss(base[5], target)
        crossloss += self.cross_entropy_loss(base[6], target)
        crossloss += self.cross_entropy_loss(base[7], target)
        crossloss += self.cross_entropy_loss(base[8], target)
        crossloss += self.cross_entropy_loss(base[9], target)
        crossloss += self.cross_entropy_loss(base[10], target)
        crossloss = TriLoss / 3 + 2 * crossloss / 8
        # crossloss = crossloss / 8
        self.scalar_summary("TriLoss", TriLoss/3)
        # self.scalar_summary("crossloss", crossloss)

        loss_sum = crossloss

        return self.get_loss(loss_sum)


class PCBLoss_v2(_Loss):
    def __init__(self, reduction='none'):
        super(PCBLoss_v2, self).__init__(reduction)
        # self.abs = ops.Abs()
        # self.reduce_mean = ops.ReduceMean()
        '''
        两种损失函数的比较
        '''
        self.l = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        # self.l = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, base, target):
        # print('输出')
        # print(base[0])
        # print('mubiao ')
        # print(target)
        y = self.l(base[3], target)
        y += self.l(base[4], target)
        y += self.l(base[5], target)
        # y += self.l(base[6], target)
        # y += self.l(base[7], target)
        # y += self.l(base[8], target)
        # y += self.l(base[9], target)
        # y += self.l(base[10], target)
        y = y / 5
        return self.get_loss(y)


if __name__ == '__main__':
    input = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
    target = Tensor([1, 1, 3, 4])
    loss = TripletLoss(batch=4, thres=Tensor(np.zeros((4, 4))))
    out = loss.construct(input, target)
    print(out)
