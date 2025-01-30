from datasets import dataloader_generator, multidataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import brain_net_transformer, neuro_detour, brain_gnn, brain_identity, bolt, graphormer, nagphormer, vanilla_model
from models.heads import Classifier, BNDecoder
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os
import numpy as np
from datetime import datetime

MODEL_BANK = {
    'neurodetour': neuro_detour.DetourTransformer,
    'neurodetourSingleFC': neuro_detour.DetourTransformerSingleFC,
    'neurodetourSingleSC': neuro_detour.DetourTransformerSingleSC,
    'bnt': brain_net_transformer.BrainNetworkTransformer,
    'braingnn': brain_gnn.Network,
    'bolt': bolt.get_BolT,
    'graphormer': graphormer.Graphormer,
    'nagphormer': nagphormer.TransformerModel,
    'transformer': vanilla_model.Transformer,
    'gcn': vanilla_model.GCN,
    'sage': vanilla_model.SAGE,
    'sgc': vanilla_model.SGC,
    'none': brain_identity.Identity
}
CLASSIFIER_BANK = {
    'mlp': nn.Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'sgc': SGConv
}
DATA_TRANSFORM = {
    'neurodetour': None,
    'neurodetourSingleFC': None,
    'neurodetourSingleSC': None,
    'bnt': None,
    'braingnn': None,
    'bolt': None,
    'graphormer': graphormer.ShortestDistance(),
    'nagphormer': nagphormer.NAGdataTransform(),
    'transformer': None,
    'gcn': None,
    'sage': None,
    'sgc': None,
    'none': None
}
ATLAS_ROI_N = {
    'AAL_116': 116,
    'Gordon_333': 333,
    'Shaefer_100': 100,
    'Shaefer_200': 200,
    'Shaefer_400': 400,
    'D_160': 160
}
DATA_CLASS_N = {
    'ukb': 2,
    'hcpa': 4,
    'hcpya': 7,
    'adni': 2,
    'oasis': 2,
    'oasis': 2,
    'ppmi': 4,
    'abide': 2,
    'neurocon': 2,
    'taowu': 2,
}
LOSS_FUNCS = {
    'y': nn.CrossEntropyLoss(),
    # 'sex': nn.CrossEntropyLoss(),
    # 'age': nn.MSELoss(), 
}
LOSS_W = {
    'y': 1,
    'sex': 1,
    'age': 1e-4,
}

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='NeuroDetour')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--models', type=str, default = 'none')
    parser.add_argument('--classifier', type=str, default = 'mlp')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 2048)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    # parser.add_argument('--dataname', type=str, default = 'ppmi')
    # parser.add_argument('--testname', type=str, default = 'None')
    parser.add_argument('--node_attr', type=str, default = 'FC')
    parser.add_argument('--adj_type', type=str, default = 'FC')
    parser.add_argument('--bold_winsize', type=int, default = 500)
    parser.add_argument('--nlayer', type=int, default = 4)
    parser.add_argument('--nhead', type=int, default = 8)
    parser.add_argument('--classifier_aggr', type=str, default = 'learn')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--fc_th', type=float, default = 0.5)
    parser.add_argument('--sc_th', type=float, default = 0.1)
    parser.add_argument('--only_dataload', action='store_true')
    parser.add_argument('--cv_fold_n', type=int, default = 10)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--decoder_layer', type=int, default = 32)
    parser.add_argument('--datanames', nargs='+', default = ['ppmi'], required=False)
    parser.add_argument('--pretrained_datanames', nargs='+', default = ['ppmi','abide','taowu','neurocon','hcpa','adni','hcpya'], required=False)
    parser.add_argument('--load_dname', type=str, default = 'hcpa', required=False)

    args = parser.parse_args()
    args.decoder = True
    print(args)
    # expdate = str(datetime.now())
    # expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    load_dname = 'ppmi'
    device = args.device
    hiddim = args.hiddim
    # nclass = DATA_CLASS_N[args.dataname]
    dataset = {'adni': None,'hcpa': None,'hcpya': None,'abide': None,'ppmi': None,'taowu': None,'neurocon': None}
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    # taccuracies = []
    # tf1_scores = []
    # tprec_scores = []
    # trec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    transform = None
    # dek, pek = 0, 0
    if args.node_attr != 'BOLD':
        input_dim = node_sz
    else:
        input_dim = args.bold_winsize
    transform = DATA_TRANSFORM[args.models]
    # testset = args.testname
    
    
    if args.decoder:
        save_mn = f'{args.models}_decoder{args.decoder_layer}'
    else:
        save_mn = f'{args.models}_mlp{args.decoder_layer}'
    
    # if args.savemodel:
    mweight_fn = f'model_weights/{save_mn}_{"-".join(args.pretrained_datanames)}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'
    assert os.path.exists(mweight_fn), mweight_fn
    # os.makedirs(mweight_fn, exist_ok=True)
    for i in range(args.cv_fold_n):
        dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset, dname_list=args.datanames,
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        train_loader, val_loader, merged_dataset, dataset = dataloaders
        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        # print(sum([p.numel() for p in model.parameters()]))
        # exit()

        pretrain_dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset, dname_list=args.pretrained_datanames,
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        pretrain_merged_dataset = pretrain_dataloaders[2]
        nclass = sum(pretrain_merged_dataset.nclass_list)
        overlap_dnames = list(np.intersect1d(args.pretrained_datanames, merged_dataset.dnames))
        if 'ppmi' in merged_dataset.dnames and 'ppmi' not in overlap_dnames:
            if 'taowu' in overlap_dnames: del overlap_dnames[overlap_dnames.index('taowu')]
            if 'neurocon' in overlap_dnames: del overlap_dnames[overlap_dnames.index('neurocon')]
            
        overlap_dtid = list(set(pretrain_merged_dataset.dname2tokenid[d] for d in overlap_dnames))
        overlap_dtoken = sum([pretrain_merged_dataset.nclass_list[tid] for tid in overlap_dtid])
        assert sum(merged_dataset.nclass_list) - overlap_dtoken >= 0, f'{sum(merged_dataset.nclass_list)} - {overlap_dtoken}'
        tokenid2dname = {}
        for d in merged_dataset.dname2tokenid:
            tid = merged_dataset.dname2tokenid[d]
            if tid not in tokenid2dname: tokenid2dname[tid] = []
            tokenid2dname[tid].append(d)
        print(nclass)
        print(merged_dataset.dname2tokenid)
        assert max(tokenid2dname.keys()) == len(tokenid2dname.keys())-1, tokenid2dname
        finetune_tokenid = []
        new_token_nclass = 0
        for tid in range(max(tokenid2dname.keys())+1):
            overlap_d = False
            for d in tokenid2dname[tid]:
                if d in overlap_dnames:
                    start_ti = sum([pretrain_merged_dataset.nclass_list[nclass_i] for nclass_i in range(pretrain_merged_dataset.dname2tokenid[d])])
                    end_ti = pretrain_merged_dataset.nclass_list[pretrain_merged_dataset.dname2tokenid[d]] + start_ti
                    f_tid = list(range(start_ti, end_ti))
                    overlap_d = True
                    break
            if not overlap_d:
                f_tid = list(range(nclass+new_token_nclass, nclass+new_token_nclass+merged_dataset.nclass_list[tid]))
                new_token_nclass += merged_dataset.nclass_list[tid]
                    
            finetune_tokenid.extend(f_tid)
        
        finetune_tokenid = torch.LongTensor(finetune_tokenid+[-3,-2,-1])
        print(finetune_tokenid)
        if args.only_dataload: exit()
        
        if not args.decoder:
            classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
        else:
            classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, finetune=True, finetune_nclass=sum(merged_dataset.nclass_list) - overlap_dtoken, finetune_tokenid=finetune_tokenid).to(device)
        print(datetime.now(), 'Load pre-trained model')        
        bb_loaded = False
        head_loaded = False
        for fn in os.listdir(mweight_fn):
            if fn.startswith(f'bb_fold{i}_{load_dname}Best_'):
                model.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'))
                bb_loaded = True
            if fn.startswith(f'head_fold{i}_{load_dname}Best_'):
                classifier.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'), strict=False)
                head_loaded = True
            if head_loaded and bb_loaded: break
        assert bb_loaded and head_loaded, f'{mweight_fn}/bb_fold{i}_{load_dname}Best_'
        print(datetime.now(), 'Done')
        
        for dname in val_loader:
            one_val_loader = val_loader[dname]
            attn, gt = eval(model, classifier, device, one_val_loader, dname=dname)
        attn = attn[..., 1:-3, :]
        print(attn.shape, gt.shape)
        fig, axes = plt.subplots((len(attn)//2+1), 2, figsize=(10,10), sharex=True, sharey=True)
        axes = axes.reshape(-1)
        avg_attn = {
            0: [],
            1: [],
            2: [],
            3: []
        }
        for bi in range(len(attn)):
            one_attn = attn[bi].detach().cpu().numpy()
            avg_attn[gt[bi].item()-1].append(one_attn)
            ax = axes[bi]
            sns.heatmap(data=one_attn, ax=ax)
            ax.set_title(f'label{gt[bi].item()-1}')
        plt.tight_layout()
        plt.savefig(f'figs/attn/{load_dname}_val-fold{i}.png')
        plt.close()
        # break
        for k in avg_attn:
            attn = np.stack(avg_attn[k]).mean(0)[k].tolist()
            attn = [str(a*10) for a in attn]
            with open(f'{load_dname}_attn_val-fold{i}_label{k}.txt', 'w') as f:
                f.write('\n'.join(attn))

def eval(model, classifier, device, loader, dname=None):
    model.eval()
    classifier.eval()
    attn = []
    gt = []
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            feat = model(batch)
            edge_index = batch.edge_index
            batchid = batch.batch
            if len(feat) == 3:  # brainGnn Selected Topk nodes
                feat, edge_index, batchid = feat
            y = classifier(feat, edge_index, batchid, return_cross_attn=True)
        
        one_y = y['y'][..., :loader.dataset.dataset.nclass_list[0]] # L x B x C
        one_gt = batch['y'][:, 0]
        print(one_y.shape)
        layeri = one_y[:, torch.arange(one_y.shape[1]), one_gt].argmax(0)
        attn.append(y['attn'][torch.arange(one_y.shape[1]), layeri].detach().cpu())
        gt.append(one_gt.detach().cpu())
    return torch.cat(attn), torch.cat(gt)

    #     pre_nclass_i = 0
    #     for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):
    #         for k in LOSS_FUNCS:
    #             if k != 'y': continue
    #             one_gt = batch[k][:, nclassi]
    #             one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
    #             if len(one_y.shape) == 3:
    #                 one_y = one_y[:, one_gt != -1].max(0)[0]
    #             else:
    #                 one_y = one_y[one_gt != -1]
    #             one_gt = one_gt[one_gt != -1]
    #             if k not in y_true_dict[nclassi]:
    #                 y_true_dict[nclassi][k] = []
    #                 y_scores_dict[nclassi][k] = []
    #             y_true_dict[nclassi][k].append(one_gt.detach().cpu())
    #             y_scores_dict[nclassi][k].append(one_y.detach().cpu())
    #         pre_nclass_i += nclass
        
    #     nclassi = len(loader.dataset.dataset.nclass_list)
    #     for k in LOSS_FUNCS:
    #         if k == 'y': continue
    #         one_gt = batch[k]
    #         one_y = y[k]
    #         if len(one_y.shape) == 3:
    #             if k == 'age':
    #                 one_y = one_y[:, one_gt != -1].mean(0)
    #             else:
    #                 one_y = one_y[:, one_gt != -1].max(0)[0]
    #         else:
    #             one_y = one_y[one_gt != -1]
    #         one_gt = one_gt[one_gt != -1]
    #         if k not in y_true_dict[nclassi]:
    #             y_true_dict[nclassi][k] = []
    #             y_scores_dict[nclassi][k] = []
    #         y_true_dict[nclassi][k].append(one_gt.detach().cpu())
    #         y_scores_dict[nclassi][k].append(one_y.detach().cpu())
    #         nclassi += 1
     
    # val_di = loader.dataset.dataset.dname2tokenid[dname] 
    # scores = {}
    # if 'y' in LOSS_FUNCS:
    #     y_true = torch.cat(y_true_dict[val_di]['y'], dim = 0).numpy()
    #     y_scores = torch.cat(y_scores_dict[val_di]['y'], dim = 0).numpy().argmax(1)
    #     acc = accuracy_score(y_true, y_scores)
    #     prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    #     scores['y'] = [acc, prec, rec, f1]
    # for di in range(len(loader.dataset.dataset.nclass_list), len(y_true_dict)):
    #     for k in y_true_dict[di]:
    #         assert k != 'y', k
    #         y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
    #         y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
    #         if k != 'age':
    #             y_true = y_true.numpy()
    #             y_scores = y_scores.numpy().argmax(1)
    #             acc = accuracy_score(y_true, y_scores)
    #             prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    #             scores[k] = [acc, prec, rec, f1]
    #         else:
    #             scores[k] = [-1, -1, -1, torch.nn.functional.mse_loss(y_scores, y_true)]

    # return scores

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()