import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


"""
proposed in this paper: [Exploring Alternatives to Softmax Function](https://arxiv.org/pdf/2011.11538.pdf)
"""


##
# version 1: use torch.autograd
class TaylorSoftmax(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        fn = torch.ones_like(x)
        denor = 1.0
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


##
# version 1: use torch.autograd
class TaylorCrossEntropyLoss(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction="mean"):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLoss(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits).log()
        loss = F.nll_loss(
            log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index
        )
        return loss

