import torch as th
from torch import nn
import math
import torch.nn.functional as F


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    out = nn.Linear(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, "pool", True):
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)