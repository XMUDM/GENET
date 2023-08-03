import dhg
import torch
import torch.nn as nn
from dhg import Hypergraph

class HGNNPConvL(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph):
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        HE = hg.v2e(X, aggr="softmax_then_sum")
        X = hg.e2v(HE,aggr="softmax_then_sum")

        return X,HE

class HGNNP(nn.Module):
    def __init__(self,
        in_channels: int,
        hid_channels: int,
        use_bn: bool = False):
        super(HGNNP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConvL(in_channels, hid_channels, use_bn=use_bn)
        )
    def forward(self,X, hg):
        global HE
        for layer in self.layers:
            X,HE = layer(X, hg)
        return X,HE
