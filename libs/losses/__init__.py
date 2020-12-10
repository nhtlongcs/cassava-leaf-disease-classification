from .classification.crossentropy import (
    BCEWithLogitsLoss,
    WeightedBCEWithLogitsLoss,
    CrossEntropyLoss,
)
from .classification.focalloss import FocalLoss
from .classification.bi_tempered_loss import BiTemperedLoss
from .segmentation.diceloss import DiceLoss
from .mixedloss import MixedLoss
