import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean, scatter_max
from scipy.optimize import linear_sum_assignment
import copy
from typing import Optional, List

class BNDecoder(nn.Module):

    def __init__(self, feat_dim, nclass, node_sz, nlayer=2, train_obj='y', dropout=0.1, head_num=8, hid_dim=768, activation="relu", normalize_before=False, obj_num=3, return_intermediate=True, finetune=False, finetune_nclass=None, finetune_tokenid=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_sz = node_sz
        # self.obj_num = obj_num
        self.feat_dim = feat_dim
        self.nclass = nclass
        self.hid_dim = hid_dim
        self.head_num = head_num
        self.train_obj = train_obj
        # self.object_query = torch.nn.Embedding(obj_num, feat_dim)#nn.Parameter(torch.randn(obj_num, self.hid_dim))
        self.object_query = torch.nn.Embedding(nclass+2+1, feat_dim)
        # self.decoder = nn.ModuleList([torch.nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(d_model=feat_dim, nhead=head_num, dim_feedforward=hid_dim, dropout=dropout, batch_first=True),
        #     num_layers=1,
        #     norm=None#nn.LayerNorm(in_channel) # None#
        # ) for _ in range(nlayer)])
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(feat_dim, head_num, hid_dim,
                                                dropout, activation, normalize_before),
            nlayer,
            nn.LayerNorm(feat_dim), # None#
            return_intermediate=return_intermediate
        )
        self.class_embed = nn.Linear(feat_dim, 1)
        # self.class_embed = nn.Linear(feat_dim, nclass)
        self.sex_embed = nn.Linear(feat_dim, 1)
        self.age_embed = nn.Linear(feat_dim, 1)
        if finetune == True:
            self.add_token(finetune_nclass, finetune_tokenid)

    def add_token(self, nclass, finetune_tokenid):
        # self.finetune_nclass = nclass
        self.finetune_tokenid = finetune_tokenid#.to(self.device)
        self.nclass += nclass
        self.finetune_query = torch.nn.Embedding(nclass, self.feat_dim)#.to(self.device)

    def forward(self, x, edge_index=None, batch=None, mask=None, target=None, return_cross_attn=False):
        x = x.reshape(-1, self.node_sz, x.shape[1]) # B x N x C_feat
        ## pre-training ######################################
        query_embed = self.object_query.weight
        if hasattr(self, 'finetune_query'):
            query_embed = torch.cat([query_embed[:-3], self.finetune_query.weight, query_embed[-3:]])
        query_embed = query_embed.unsqueeze(0).repeat(len(x), 1, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, x, memory_key_padding_mask=mask, query_pos=query_embed, return_cross_attn=return_cross_attn) # B X N X C
        if return_cross_attn:
            hs, attn = hs
        ###########################################
        # logits = self.class_embed(hs[..., :self.nclass, :]).squeeze(-1)
        # sex = self.sex_embed(hs[..., -3:-1, :]).squeeze(-1)
        # age = self.age_embed(hs[..., -1, :])
        ##############################################
        logits = self.class_embed(hs).squeeze(-1)
        sex = logits[..., -3:-1]
        age = logits[..., -1:]
        logits = logits[..., :self.nclass]
        ############################################
        if hasattr(self, 'finetune_query'):
            logits = logits[..., self.finetune_tokenid]
            if return_cross_attn:
                attn = attn[..., self.finetune_tokenid,:]
        if not return_cross_attn:
            return {self.train_obj: logits}
            # return {self.train_obj: logits, 'sex': sex, 'age': age}
        else:
            # return {self.train_obj: logits, 'sex': sex, 'age': age, 'attn': attn}
            return {self.train_obj: logits, 'attn': attn}


class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, node_sz, nlayer=1, aggr='learn', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlayer = nlayer
        self.net = []
        for _ in range(nlayer):
            self.net.append(net(feat_dim, feat_dim))
            self.net.append(nn.LeakyReLU())
        
        self.net.append(net(feat_dim, nclass+3))
        self.net = nn.ModuleList(self.net)
        if isinstance(self.net[0], MessagePassing):
            self.nettype = 'gnn'
        else:
            self.nettype = 'mlp'
        self.aggr = aggr
        if aggr == 'learn':
            self.pool = nn.Sequential(nn.Linear(node_sz, 1), nn.LeakyReLU())
        elif aggr == 'mean':
            self.pool = scatter_mean
        elif aggr == 'max':
            self.pool = scatter_max
        self.nclass = nclass
    
    def forward(self, x, edge_index, batch):
        if self.nettype == 'gnn':
            for i in range(0, (len(self.net)-1)//2, 2):
                x = self.net[i](x, edge_index)
                x = self.net[i+1](x)
            x = self.net[-1](x, edge_index)
        else:
            for i in range(0, (len(self.net)-1)//2, 2):
                x = self.net[i](x)
                x = self.net[i+1](x)
            x = self.net[-1](x)
    
        if self.aggr == 'learn':
            x = self.pool(x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2))[..., 0]
        else:
            if self.aggr == 'max': 
                x = x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2).max(-1)[0]
            else:
                x = self.pool(x, batch, dim=0)
        return {'y': x[..., :self.nclass], 'sex': x[..., -3:-1], 'age': x[..., -1:]}


class HungarianLoss(nn.Module):

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):

        super().__init__()
        self.cost_class = cost_class
        # self.cost_bbox = cost_bbox
        # self.cost_giou = cost_giou
        # assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets): # bs x Nobj x Ncls, bs x [Nann]
        bs, num_queries = outputs.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs.flatten(0, 1).softmax(-1)  # B*Nobj x Ncls

        # Also concat the target labels and boxes
        vg_obj_label_mask = targets['vg_obj_cls']!=0
        sizes = (vg_obj_label_mask).sum(-1).tolist() # B
        tgt_ids = targets['vg_obj_cls'][vg_obj_label_mask].long() # B*Nann x 1

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] # B*Nobj x B*Nann
        # Final cost matrix
        C = self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).detach().cpu().split(sizes, -1)

        tgt_queries = torch.zeros(bs, num_queries).long()
        for batchi, c in enumerate(C):
            indices = linear_sum_assignment(c[batchi].numpy())
            tgt_queries[batchi, indices[0]] = torch.from_numpy(indices[1])

        loss = F.cross_entropy(outputs.flatten(0, 1), tgt_queries.flatten(0, 1).to(outputs.device))
        return loss 


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_cross_attn=False):
        output = tgt

        intermediate = []
        attn = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_cross_attn=return_cross_attn)
            if return_cross_attn:
                attn.append(output[1])
                output = output[0]
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            
            if not return_cross_attn:
                return torch.stack(intermediate)
            else:
                return torch.stack(intermediate), torch.stack(attn, dim=1) # B x L x P x N
        if not return_cross_attn:
            return output#.unsqueeze(0)
        else:
            return output, torch.stack(attn, dim=1) # B x L x P x N

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_cross_attn=False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_cross_attn=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not return_cross_attn:
            return tgt
        else:
            return tgt, cross_attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_cross_attn=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_cross_attn)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
