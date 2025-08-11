from datasets import dataloader_generator, multidataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from models import brain_net_transformer, neuro_detour, brain_gnn, brain_identity, bolt, graphormer, nagphormer, vanilla_model
from models.heads import Classifier, BNDecoder
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch, math
import argparse, os, json
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
    'sz-diana': 2,
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

def main():
    parser = argparse.ArgumentParser(description='NeuroDetour')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--models', type=str, default = 'none')
    parser.add_argument('--classifier', type=str, default = 'mlp')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 2048)
    parser.add_argument('--lr', type=float, default = 0.00001)
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
    parser.add_argument('--device', type=str, default = 'cuda:2')
    parser.add_argument('--fc_th', type=float, default = 0.5)
    parser.add_argument('--sc_th', type=float, default = 0.1)
    parser.add_argument('--only_dataload', action='store_true')
    parser.add_argument('--cv_fold_n', type=int, default = 10)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--decoder_layer', type=int, default = 32)
    parser.add_argument('--datanames', nargs='+', default = ['adni','abide','ppmi','taowu','neurocon'], required=False)
    parser.add_argument('--pretrained_datanames', nargs='+', default = ['ppmi','abide','taowu','neurocon','hcpa', 'adni','hcpya'], required=False)
    parser.add_argument('--load_dname', type=str, default = 'hcpa', required=False)
    parser.add_argument('--few_shot', type=float, default = 1)
    parser.add_argument('--force_2class', action='store_true')


    args = parser.parse_args()
    # args.decoder = True
    print(args)
    # expdate = str(datetime.now())
    # expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    load_dname = args.load_dname#'hcpa'
    device = args.device
    hiddim = args.hiddim
    # nclass = DATA_CLASS_N[args.dataname]
    dataset_dict = {dn: None for dn in args.datanames}
    pretrain_dataset_dict = {dn: None for dn in args.pretrained_datanames}
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    auc_scores_dict = {}
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
    # for i in range(args.cv_fold_n):
    _nfold = {
        'ukb': 5,
        'hcpa': 5,
        'hcpya': 5,
        'adni': 5,
        'oasis': 5,
        'ppmi': 10,
        'abide': 10,
        'neurocon': 10,
        'taowu': 10,
        'sz-diana': 10,
    }
    # foldi = 0
    i = 0
    # for foldi in range(5):
    # while foldi < 5 and i < _nfold[args.datanames[0]]:
    while i < _nfold[args.datanames[0]]:
        dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset_dict, dname_list=args.datanames, few_shot=args.few_shot,
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        train_loader, val_loader, merged_dataset, dataset_dict = dataloaders
        vald_all_pass = True
        for dname in val_loader:
            uni_label = torch.cat([data['y'] for data in val_loader[dname]]).unique()
            if args.force_2class: uni_label = uni_label[uni_label<=2]
            print('uni_label', uni_label)
            if len(uni_label) == 1: 
                vald_all_pass = False
                break
        if not vald_all_pass: continue
        # foldi += 1
        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        # print(sum([p.numel() for p in model.parameters()]))
        # exit()
        meta_info_fn = 'model_weights/' + '-'.join(args.pretrained_datanames) + '_meta.json'
        if not os.path.exists(meta_info_fn):
            pretrain_dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=pretrain_dataset_dict, dname_list=args.pretrained_datanames,
                                                                    node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
                                                                    fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
            pretrain_merged_dataset, pretrain_dataset_dict = pretrain_dataloaders[2:]
            pretrain_nclass_list = pretrain_merged_dataset.nclass_list
            pretrain_dname2tokenid = pretrain_merged_dataset.dname2tokenid
            with open(meta_info_fn, 'w') as file:
                json.dump({'nclass_list': pretrain_nclass_list, 'dname2tokenid': pretrain_dname2tokenid}, file)
        else:
            with open(meta_info_fn, 'r') as file:
                meta_data = json.load(file)
            pretrain_nclass_list = meta_data['nclass_list']
            pretrain_dname2tokenid = meta_data['dname2tokenid']


        nclass = sum(pretrain_nclass_list)

        overlap_dnames = list(np.intersect1d(args.pretrained_datanames, merged_dataset.dnames))
        if 'ppmi' in merged_dataset.dnames and 'ppmi' not in overlap_dnames:
            if 'taowu' in overlap_dnames: del overlap_dnames[overlap_dnames.index('taowu')]
            if 'neurocon' in overlap_dnames: del overlap_dnames[overlap_dnames.index('neurocon')]
            
        overlap_dtid = list(set(pretrain_dname2tokenid[d] for d in overlap_dnames))
        overlap_dtoken = sum([pretrain_nclass_list[tid] for tid in overlap_dtid])
        # assert sum(merged_dataset.nclass_list) - overlap_dtoken >= 0, f'{sum(merged_dataset.nclass_list)} - {overlap_dtoken}'
        new_token_num = max(sum(merged_dataset.nclass_list) - overlap_dtoken, 0)
        tokenid2dname = {}
        for d in merged_dataset.dname2tokenid:
            tid = merged_dataset.dname2tokenid[d]
            if tid not in tokenid2dname: tokenid2dname[tid] = []
            tokenid2dname[tid].append(d)
        print(merged_dataset.dname2tokenid)
        assert max(tokenid2dname.keys()) == len(tokenid2dname.keys())-1, tokenid2dname
        finetune_tokenid = []
        new_token_nclass = 0
        for tid in range(max(tokenid2dname.keys())+1):
            overlap_d = False
            for d in tokenid2dname[tid]:
                if d in overlap_dnames:
                    start_ti = sum([pretrain_nclass_list[nclass_i] for nclass_i in range(pretrain_dname2tokenid[d])])
                    end_ti = pretrain_nclass_list[pretrain_dname2tokenid[d]] + start_ti
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
            classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, finetune=True, finetune_nclass=new_token_num, finetune_tokenid=finetune_tokenid).to(device)
        print(datetime.now(), 'Load pre-trained model')        
        bb_loaded = False
        head_loaded = False
        for fn in os.listdir(mweight_fn):
            if fn.startswith(f'bb_fold{min(i,4)}_{load_dname}Best_'):
                model.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'))
                bb_loaded = True
            if fn.startswith(f'head_fold{min(i,4)}_{load_dname}Best_'):
                classifier.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'), strict=False)
                head_loaded = True
            if head_loaded and bb_loaded: break
        assert bb_loaded and head_loaded, f'{mweight_fn}/bb_fold{min(i,4)}_{load_dname}Best_'
        print(datetime.now(), 'Done')
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        
        best_f1 = {}
        best_acc = {}
        best_prec = {}
        best_rec = {}
        best_auc = {}
        # patience = {}
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            print(datetime.now(), 'train start')
            train(model, classifier, device, train_loader, optimizer, epoch, force_2class=args.force_2class)
            print(datetime.now(), 'train done, test start')
            for dname in val_loader:
                if dname not in best_f1:
                    best_f1[dname] = {}
                    best_acc[dname] = {}
                    best_prec[dname] = {}
                    best_rec[dname] = {}
                    best_auc[dname] = {}
                one_val_loader = val_loader[dname]
                # acc, prec, rec, f1 = eval(model, classifier, device, one_val_loader, dname=dname)
                scores = eval(model, classifier, device, one_val_loader, dname=dname, force_2class=args.force_2class)
                print(datetime.now(), 'test done')
                log = f'Dataset: {dname} [Accuracy, F1 Score]:'
                for k in scores:
                    acc, prec, rec, f1, auc_score = scores[k]
                    if scores[k][0] == -1:
                        f1 = -1*f1

                    log += f'({k}) [{acc:.6f},  {f1:.6f}], {auc_score:.6f}\t'
                    if k not in best_f1[dname]:
                        best_f1[dname][k] = -torch.inf
                        best_acc[dname][k] = -torch.inf
                        best_prec[dname][k] = -torch.inf
                        best_rec[dname][k] = -torch.inf
                        best_auc[dname][k] = -torch.inf
                    
                    # if f1 >= best_f1[dname][k]:
                    if f1 + auc_score >= best_auc[dname][k] + best_f1[dname][k]:
                        best_f1[dname][k] = f1
                        best_acc[dname][k] = acc
                        best_prec[dname][k] = prec
                        best_rec[dname][k] = rec
                        best_auc[dname][k] = auc_score
                        # if args.savemodel:
                        #     torch.save(model.state_dict(), f'{mweight_fn}/bb_fold{i}_{dname}Best-{k}_{expdate}.pt')
                        #     torch.save(classifier.state_dict(), f'{mweight_fn}/head_fold{i}_{dname}Best-{k}_{expdate}.pt')
                
                print(log)
        
        for dname in best_acc:
            log = f'Dataset: {dname} [Accuracy, F1 Score, Prec, Rec, AUC]:'
            for k in best_acc[dname]:
                if dname not in accuracies_dict: accuracies_dict[dname] = {k: []}
                if dname not in f1_scores_dict: f1_scores_dict[dname] = {k: []}
                if dname not in prec_scores_dict: prec_scores_dict[dname] = {k: []}
                if dname not in rec_scores_dict: rec_scores_dict[dname] = {k: []}
                if dname not in auc_scores_dict: auc_scores_dict[dname] = {k: []}
                if k not in accuracies_dict[dname]:
                    accuracies_dict[dname][k] = []
                    f1_scores_dict[dname][k] = []
                    prec_scores_dict[dname][k] = []
                    rec_scores_dict[dname][k] = []
                    auc_scores_dict[dname][k] = []
                accuracies_dict[dname][k].append(best_acc[dname][k])
                f1_scores_dict[dname][k].append(best_f1[dname][k])
                prec_scores_dict[dname][k].append(best_prec[dname][k])
                rec_scores_dict[dname][k].append(best_rec[dname][k])
                auc_scores_dict[dname][k].append(best_auc[dname][k])
                log += f'({k}) [{best_acc[dname][k]}, {best_f1[dname][k]}, {best_prec[dname][k]}, {best_rec[dname][k]}, {best_auc[dname][k]}], \t'
            print(log)

        i += 1
    # Calculate mean and standard deviation of evaluation metrics
    for dname in accuracies_dict:
        for k in accuracies_dict[dname]:
            accuracies = accuracies_dict[dname][k]
            f1_scores = f1_scores_dict[dname][k]
            prec_scores = prec_scores_dict[dname][k]
            rec_scores = rec_scores_dict[dname][k]
            auc_scores = auc_scores_dict[dname][k]
            mean_accuracy = sum(accuracies) / len(accuracies)
            std_accuracy = torch.std(torch.tensor(accuracies))
            mean_f1_score = sum(f1_scores) / len(f1_scores)
            std_f1_score = torch.std(torch.tensor(f1_scores))
            mean_prec_score = sum(prec_scores) / len(prec_scores)
            std_prec_score = torch.std(torch.tensor(prec_scores))
            mean_rec_score = sum(rec_scores) / len(rec_scores)
            std_rec_score = torch.std(torch.tensor(rec_scores))
            mean_auc_score = sum(auc_scores) / len(auc_scores)
            std_auc_score = torch.std(torch.tensor(auc_scores))
            print(f'Dataset: {dname} ({k})')
            print(f'Mean Accuracy: {mean_auc_score}, Std Accuracy: {std_auc_score}')
            print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
            print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
            print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
            # print(f'Mean auc Score: {mean_auc_score}, Std rec Score: {std_auc_score}')

        
def train(model, classifier, device, loader, optimizer, epoch, force_2class=False):
    model.train()
    classifier.train()
    losses = []
    y_true_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    y_scores_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    layeri_selected = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    # loss_fn = nn.CrossEntropyLoss()
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        feat = model(batch)
        edge_index = batch.edge_index
        batchid = batch.batch
        if len(feat) == 3:  # brainGnn Selected Topk nodes
            feat, edge_index, batchid = feat
        y = classifier(feat, edge_index, batchid)
        loss = 0
        for k in LOSS_FUNCS:
            if k == 'y':
                pre_nclass_i = 0
                for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):
                    if k not in layeri_selected[nclassi]: 
                        layeri_selected[nclassi][k] = []
                    # _nclass = nclass
                    # if force_2class: _nclass = 3
                    one_gt = batch[k][:, nclassi]
                    one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                    if len(one_y.shape) == 3:
                        one_y = one_y[:, one_gt != -1]                    
                        one_gt = one_gt[one_gt != -1]
                        if epoch > 5:
                            one_yi = torch.arange(one_y.shape[1])
                            layeri = one_y[:, one_yi, one_gt].argmax(0)
                            one_y = one_y[layeri, one_yi]
                            layeri_selected[nclassi][k].append(layeri.detach().cpu())
                        else:
                            one_y = one_y.mean(0)
                    else:
                        one_y = one_y[one_gt != -1]
                        one_gt = one_gt[one_gt != -1]
                    if force_2class:
                        # print(one_y.shape)
                        one_y = one_y[one_gt <= 2]
                        one_gt = one_gt[one_gt <= 2]
                        # print(one_y.shape)
                        # exit()
                    loss += LOSS_W[k]*LOSS_FUNCS[k](one_y, one_gt)
                    pre_nclass_i += nclass
            else: # sex or age
                one_gt = batch[k]
                one_y = y[k]
                if len(one_y.shape) == 3:
                    one_y = one_y[:, one_gt != -1]                    
                    one_gt = one_gt[one_gt != -1]
                    if epoch > 5 and k != 'age':
                        one_yi = torch.arange(one_y.shape[1])
                        layeri = one_y[:, one_yi, one_gt].argmax(0)
                        one_y = one_y[layeri, one_yi]
                    else:
                        one_y = one_y.mean(0)
                else:
                    one_y = one_y[one_gt != -1]
                    one_gt = one_gt[one_gt != -1]

                loss += LOSS_W[k]*LOSS_FUNCS[k](one_y, one_gt)
            # print(k, y[k].shape, loss)
        
        # exit()
        if hasattr(model, 'loss'):
            loss = loss + model.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        pre_nclass_i = 0
        for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):   
            # _nclass = nclass
            # if force_2class: _nclass = 3
            for k in LOSS_FUNCS:
                if k != 'y': continue
                one_gt = batch[k][:, nclassi]
                one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                if len(one_y.shape) == 3:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
                else:
                    one_y = one_y[one_gt != -1]
                one_gt = one_gt[one_gt != -1]
                if k not in y_true_dict[nclassi]: 
                    y_true_dict[nclassi][k] = []
                    y_scores_dict[nclassi][k] = []
                y_true_dict[nclassi][k].append(one_gt.detach().cpu())
                y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            pre_nclass_i += nclass
        
        nclassi = len(loader.dataset.dataset.nclass_list)
        for k in LOSS_FUNCS:
            if k == 'y': continue
            one_gt = batch[k]
            one_y = y[k]
            if len(one_y.shape) == 3:
                if k == 'age':
                    one_y = one_y[:, one_gt != -1].mean(0)
                else:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
            else:
                one_y = one_y[one_gt != -1]
            one_gt = one_gt[one_gt != -1]
            if k not in y_true_dict[nclassi]: 
                y_true_dict[nclassi][k] = []
                y_scores_dict[nclassi][k] = []
            y_true_dict[nclassi][k].append(one_gt.detach().cpu())
            y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            nclassi += 1
    # print([_y_true_dict.keys() for _y_true_dict in y_true_dict])
    logs = [f'Train loss: {np.mean(losses):.6f}']
    for di, dname in enumerate(loader.dataset.dataset.dnames):
        di = loader.dataset.dataset.dname2tokenid[dname]
        for k in y_true_dict[di]:
            assert k == 'y', f'{k},{di},{dname},{len(y_true_dict)}'
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu() 
            layeri = torch.cat(layeri_selected[di][k], dim = 0).detach().cpu().numpy() if len(layeri_selected[di][k]) > 0 else []
            np.save(f'{dname}_lcm_layeri.npy', {'layeri_selected': layeri, 'gt': y_true, 'pred': y_scores.argmax(1)}, allow_pickle=True)
            if force_2class:
                y_scores = y_scores[y_true <= 2]
                y_true = y_true[y_true <= 2]
            y_true = y_true.numpy()
            y_scores = y_scores.numpy().argmax(1)
            acc = accuracy_score(y_true, y_scores)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
            logs.append(f'{dname}-{k}-Accuracy: {acc:.6f}')
            
    for di in range(len(loader.dataset.dataset.nclass_list), len(y_true_dict)):
        dname = 'All'
        for k in y_true_dict[di]:
            assert k != 'y', k
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
            if k != 'age':
                y_true = y_true.numpy()
                y_scores = y_scores.numpy().argmax(1)
                acc = accuracy_score(y_true, y_scores)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
                logs.append(f'{dname}-{k}-Accuracy: {acc:.6f}')
            else:
                logs.append(f'{dname}-{k}-MSE: {torch.nn.functional.mse_loss(y_scores, y_true):.6f}')
    
    print(', '.join(logs))

def eval(model, classifier, device, loader, dname=None, force_2class=False):
    model.eval()
    classifier.eval()
    y_true = [[]]
    y_scores = [[]]
    y_true_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    y_scores_dict = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    layeri_selected = [{} for i in range(len(loader.dataset.dataset.nclass_list)+len(LOSS_FUNCS)-1)]
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            feat = model(batch)
            edge_index = batch.edge_index
            batchid = batch.batch
            if len(feat) == 3:  # brainGnn Selected Topk nodes
                feat, edge_index, batchid = feat
            y = classifier(feat, edge_index, batchid)

        pre_nclass_i = 0
        for nclassi, nclass in enumerate(loader.dataset.dataset.nclass_list):
            # _nclass = nclass
            # if force_2class: _nclass = 3
            for k in LOSS_FUNCS:
                if k != 'y': continue
                one_gt = batch[k][:, nclassi]
                one_y = y[k][..., pre_nclass_i:pre_nclass_i+nclass]
                if len(one_y.shape) == 3:
                    layeri = one_y[:, one_gt != -1].argmax(0) # Nlayer x Nbatch x Ntoken
                    one_y = one_y[:, one_gt != -1].max(0)[0]
                    layeri = layeri[torch.arange(len(one_y)).to(one_y.device), one_y.argmax(1)]
                else:
                    one_y = one_y[one_gt != -1]
                    layeri = torch.zeros(len(one_y)).to(one_y.device)

                one_gt = one_gt[one_gt != -1]
                if k not in y_true_dict[nclassi]:
                    y_true_dict[nclassi][k] = []
                    y_scores_dict[nclassi][k] = []
                    layeri_selected[nclassi][k] = []
                y_true_dict[nclassi][k].append(one_gt.detach().cpu())
                y_scores_dict[nclassi][k].append(one_y.detach().cpu())
                layeri_selected[nclassi][k].append(layeri.detach().cpu())
            pre_nclass_i += nclass
        
        nclassi = len(loader.dataset.dataset.nclass_list)
        for k in LOSS_FUNCS:
            if k == 'y': continue
            one_gt = batch[k]
            one_y = y[k]
            if len(one_y.shape) == 3:
                if k == 'age':
                    one_y = one_y[:, one_gt != -1].mean(0)
                else:
                    one_y = one_y[:, one_gt != -1].max(0)[0]
            else:
                one_y = one_y[one_gt != -1]
            one_gt = one_gt[one_gt != -1]
            if k not in y_true_dict[nclassi]:
                y_true_dict[nclassi][k] = []
                y_scores_dict[nclassi][k] = []
            y_true_dict[nclassi][k].append(one_gt.detach().cpu())
            y_scores_dict[nclassi][k].append(one_y.detach().cpu())
            nclassi += 1
     
    val_di = loader.dataset.dataset.dname2tokenid[dname] 
    scores = {}
    if 'y' in LOSS_FUNCS:
        y_true = torch.cat(y_true_dict[val_di]['y'], dim = 0).numpy()
        y_scores = torch.cat(y_scores_dict[val_di]['y'], dim = 0).softmax(1).numpy()
        layeri = torch.cat(layeri_selected[val_di]['y'], dim = 0).numpy()
        # np.save(f'{dname}_lcm_layeri.npy', {'layeri_selected': layeri, 'gt': y_true, 'pred': y_scores.argmax(1)}, allow_pickle=True)
        if force_2class:
            y_scores = y_scores[y_true <= 2]
            y_true = y_true[y_true <= 2]
        acc = accuracy_score(y_true, y_scores.argmax(1))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores.argmax(1), average='weighted')
        fpr, tpr, thresholds = roc_curve(y_true-1, y_scores[:, 2], pos_label=1)
        auc_score = auc(fpr, tpr)
        scores['y'] = [acc, prec, rec, f1, auc_score]
    for di in range(len(loader.dataset.dataset.nclass_list), len(y_true_dict)):
        for k in y_true_dict[di]:
            assert k != 'y', k
            y_true = torch.cat(y_true_dict[di][k], dim = 0).detach().cpu()
            y_scores = torch.cat(y_scores_dict[di][k], dim = 0).detach().cpu()
            if k != 'age':
                y_true = y_true.numpy()
                y_scores = y_scores.numpy().argmax(1)
                acc = accuracy_score(y_true, y_scores)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
                scores[k] = [acc, prec, rec, f1]
            else:
                scores[k] = [-1, -1, -1, torch.nn.functional.mse_loss(y_scores, y_true)]

    return scores

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()