import torch.nn as nn

from networks.caml import CAML
from networks.kim_cnn import KimCNN
from networks.kim_cnn_v2 import KimCNNv2
from networks.xml_cnn import XMLCNN


def get_weight_init_func(config):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            getattr(nn.init, config.weight_init + '_')(m.weight)
    return init_weights
