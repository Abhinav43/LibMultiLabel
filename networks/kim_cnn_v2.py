import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseModel


class KimCNNv2(BaseModel):
    def __init__(self, config, embed_vecs):
        super(KimCNNv2, self).__init__(config, embed_vecs)

        self.filter_sizes = config.filter_sizes
        emb_dim = embed_vecs.shape[1]
        num_filter_maps = config.num_filter_maps
        strides = config.strides

        self.convs = nn.ModuleList()

        for filter_size, stride in zip(self.filter_sizes, strides):
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=num_filter_maps,
                kernel_size=(filter_size, emb_dim),
                stride=(stride, emb_dim))
            self.convs.append(conv)
        conv_output_size = num_filter_maps * len(self.filter_sizes)

        self.dropout2 = nn.Dropout(config.dropout2)
        self.linear1 = nn.Linear(conv_output_size, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.view(h.shape[0], 1, h.shape[1], h.shape[2]) # (batch_size, 1, length, embed_dim)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, H, 1)
            h_sub = h_sub.view(h_sub.shape[0], h_sub.shape[1], -1) # (batch_size, num_filter, H)
            h_sub = F.max_pool1d(h_sub, kernel_size=h_sub.size()[2]) # (batch_size, num_filter, 1)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter)
            h_list.append(h_sub)

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        h = F.relu(h) # (batch_size, N * num_filter)

        # linear output
        h = F.relu(self.linear1(h))
        h = self.dropout2(h)
        h = self.linear2(h)
        return {'logits': h}
