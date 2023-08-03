import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.utils import sparse_dropout


class GCFSignal(nn.Module):
    def __init__(
            self, num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers


    def forward(self, u_embs,i_embs,ui_bigraph):
        all_embs = torch.cat([u_embs, i_embs], dim=0)
        embs_list = [all_embs]
        # if self.train():
        #     ui_bigraph = sparse_dropout(ui_bigraph, 0.1)
        for _ in range(self.num_layers):
            all_embs = ui_bigraph.smoothing_with_GCN(all_embs)
            embs_list.append(all_embs)
        embs = torch.stack(embs_list, dim=1)
        embs = torch.mean(embs, dim=1)
        return embs


class LG2SeqSignal(nn.Module):
    def __init__(
            self, emb_dim=64,num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.encoder = GCFSignal(num_layers)
        self.docoder = nn.ModuleList()
        for i in range(num_layers):
            self.docoder.append(nn.GRU(emb_dim,emb_dim))

    def forward(self,u_embs,i_embs,ui_bigraph,user_seq):
        emb = self.encoder(u_embs,i_embs,ui_bigraph) #(n,emb_dim)
        # user_seqs是一个batch的用户序列，每个序列的长度不一样，所以需要padding
        de_emb = emb[user_seq] #(batch_size,seq_len,emb_dim)
        # print(de_emb.shape)
        de_emb = de_emb.transpose(1,0) #(seq_len,batch_size,emb_dim)
        # print(de_emb.shape)
        for decoder in self.docoder:
            o,h = decoder(de_emb)
            de_emb = h
        # o,h = self.docoder(de_emb) #
        # print(o.shape,h.shape)
        h = h.squeeze(0)
        return h,emb

class CFSignal(nn.Module):
    def __init__(
            self, emb_dim=64,num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.u_layers = nn.ModuleList()
        self.i_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.u_layers.append(nn.Linear(emb_dim, emb_dim))
            self.i_layers.append(nn.Linear(emb_dim, emb_dim))


    def forward(self, u_embs,i_embs):
        emb_list = [torch.cat([u_embs, i_embs], dim=0)]

        for i in range(self.num_layers-1):
            u_embs = self.u_layers[i](u_embs)
            i_embs = self.i_layers[i](i_embs)
            emb_list.append(torch.cat([u_embs, i_embs], dim=0))
        embs = torch.stack(emb_list, dim=1)
        embs = torch.mean(embs, dim=1)
        # u_embs, i_embs = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return embs




