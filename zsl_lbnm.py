from datasets import dataloader_generator, multidataloader_generator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
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
import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    parser.add_argument('--dataname', type=str, default = 'sz-diana')
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
    parser.add_argument('--device', type=str, default = 'cuda:5')
    parser.add_argument('--fc_th', type=float, default = 0.5)
    parser.add_argument('--sc_th', type=float, default = 0.1)
    parser.add_argument('--only_dataload', action='store_true')
    parser.add_argument('--cv_fold_n', type=int, default = 10)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--decoder_layer', type=int, default = 32)
    # parser.add_argument('--datanames', nargs='+', default = ['abide'], required=False)
    # parser.add_argument('--pretrained_datanames', nargs='+', default = ['ppmi','abide','taowu','neurocon','hcpa','adni','hcpya'], required=False) # adni ppmi taowu neurocon hcpa hcpya
    # parser.add_argument('--pretrained_datanames', nargs='+', default = ['adni','ppmi','taowu','neurocon','hcpa','hcpya'], required=False) # adni ppmi taowu neurocon hcpa hcpya
    # parser.add_argument('--pretrained_datanames', nargs='+', default = ['adni','abide','taowu','neurocon','hcpa','hcpya'], required=False) # adni ppmi taowu neurocon hcpa hcpya
    parser.add_argument('--pretrained_datanames', nargs='+', default = ['hcpa','hcpya'], required=False) # adni ppmi taowu neurocon hcpa hcpya
    # parser.add_argument('--pretrained_datanames', nargs='+', default = ['ppmi','abide','taowu','neurocon','hcpa','hcpya'], required=False) # adni ppmi taowu neurocon hcpa hcpya
    parser.add_argument('--load_dname', type=str, default = 'hcpa', required=False)
    parser.add_argument('--few_shot', type=float, default = 0.01)


    args = parser.parse_args()
    args.decoder = True
    print(args)
    # expdate = str(datetime.now())
    # expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    load_dname = args.load_dname#'hcpa'
    device = args.device
    hiddim = args.hiddim
    # nclass = DATA_CLASS_N[args.dataname]
    # dataset_dict = {dn: None for dn in args.datanames}
    pretrain_dataset_dict = {dn: None for dn in args.pretrained_datanames}
    # Initialize lists to store evaluation metrics
    accuracies_dict = {'y': []}
    f1_scores_dict = {'y': []}
    prec_scores_dict = {'y': []}
    rec_scores_dict = {'y': []}
    auc_scores_dict = {'y': []}
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
    dataset = None
    mweight_fn = f'model_weights/{save_mn}_{"-".join(args.pretrained_datanames)}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'
    assert os.path.exists(mweight_fn), mweight_fn
    # os.makedirs(mweight_fn, exist_ok=True)
    # for i in range(args.cv_fold_n):
    # for i in range(5):
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
    foldi = 0
    i = 0
    # for foldi in range(5):
    while foldi < 5 and i+1 < _nfold[args.dataname]:
        i += 1
        # dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=dataset_dict, dname_list=args.datanames, few_shot=args.few_shot,
        #                                                          node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
        #                                                          fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        # train_loader, val_loader, merged_dataset, dataset_dict = dataloaders
        dataloaders = dataloader_generator(batch_size=args.batch_size, nfold=i, dataset=dataset, few_shot=args.few_shot,
                                                            node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, dname=args.dataname,
                                                            fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        if args.only_dataload: exit()
        if len(dataloaders) == 3:
            train_loader, val_loader, dataset = dataloaders
        else:
            train_loader, val_loader, dataset, test_loader, testset = dataloaders
        # uni_label = torch.cat([data['y'] for data in train_loader]).unique()
        # uni_label = uni_label[uni_label<=1]
        # print('uni_label', uni_label)
        # if len(uni_label) == 1: continue
        uni_label = torch.cat([data['y'] for data in val_loader]).unique()
        uni_label = uni_label[uni_label<=1]
        print('uni_label', uni_label)
        if len(uni_label) == 1: continue
        foldi += 1    

        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        # print(sum([p.numel() for p in model.parameters()]))
        # exit()

        pretrain_dataloaders = multidataloader_generator(batch_size=args.batch_size, nfold=i, datasets=pretrain_dataset_dict, dname_list=args.pretrained_datanames,
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, #testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        pretrain_traindata_loader, _, pretrain_merged_dataset, pretrain_dataset_dict = pretrain_dataloaders
        nclass = sum(pretrain_merged_dataset.nclass_list)

        if args.only_dataload: exit()
        
        if not args.decoder:
            classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
        else:
            classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, finetune=False).to(device)
        print(datetime.now(), 'Load pre-trained model')        
        bb_loaded = False
        head_loaded = False
        for fn in os.listdir(mweight_fn):
            if fn.startswith(f'bb_fold{min(i, 4)}_{load_dname}Best_'):
                model.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'))
                bb_loaded = True
            if fn.startswith(f'head_fold{min(i, 4)}_{load_dname}Best_'):
                classifier.load_state_dict(torch.load(f'{mweight_fn}/{fn}', map_location='cpu'), strict=False)
                head_loaded = True
            if head_loaded and bb_loaded: break
        assert bb_loaded and head_loaded, f'{mweight_fn}/bb_fold{min(i, 4)}_{load_dname}Best_'
        print(datetime.now(), 'Done')

        attn_path = f"beca_map/attn_fold{i}_{'-'.join(args.pretrained_datanames)}.pt"
        if not os.path.exists(attn_path):
            all_attn, all_y = eval(model, classifier, device, pretrain_traindata_loader, dname='pretrain')

            # print(all_y)
            ## Convert all_y to one-hot
            train_y = torch.zeros(len(all_y), all_attn.shape[1], device=all_y.device)
            cumsum_ncls = [0] + list(np.cumsum(pretrain_traindata_loader.dataset.dataset.nclass_list))[:-1]
            for nclsi, ncls in enumerate(cumsum_ncls):
                train_y[torch.arange(len(train_y)), ncls+all_y[:, nclsi]] = 1
            
            torch.save([all_attn, train_y], attn_path)
        all_attn, train_y = torch.load(attn_path, map_location=device)
        
        acc, f1, prec, rec, auc_score = eval(model, classifier, device, val_loader, dname=args.dataname, train_attn=all_attn, train_y=train_y, search_train_loader=train_loader if args.few_shot > 0 else None)
        print(f'Acc, F1, Prec, Rec, AUC: [{acc} , {f1} , {prec} , {rec} , {auc_score}]')
        accuracies_dict['y'].append(acc)
        f1_scores_dict['y'].append(f1)
        prec_scores_dict['y'].append(prec)
        rec_scores_dict['y'].append(rec)
        auc_scores_dict['y'].append(auc_score)
        continue
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            # print(datetime.now(), 'train start')
            # train(model, classifier, device, train_loader, optimizer, epoch)
            # print(datetime.now(), 'train done, test start')
            
            scores = eval(model, classifier, device, val_loader)
            print(datetime.now(), 'test done')
            log = f'Dataset: {args.dataname} [Accuracy, F1 Score]:'
            for k in scores:
                acc, prec, rec, f1 = scores[k]
                if scores[k][0] == -1:
                    f1 = -1*f1

                log += f'({k}) [{acc:.6f},  {f1:.6f}], \t'
                if k not in best_f1:
                    best_f1[k] = -torch.inf
                    best_acc[k] = -torch.inf
                    best_prec[k] = -torch.inf
                    best_rec[k] = -torch.inf
                
                if f1 > best_f1[k]:
                    best_f1[k] = f1
                    best_acc[k] = acc
                    best_prec[k] = prec
                    best_rec[k] = rec 
                    if k == args.train_obj:
                        patience = 0
                        
                    # if args.savemodel:
                    #     torch.save(model.state_dict(), f'{mweight_fn}/bb_fold{i}_{dname}Best-{k}_{expdate}.pt')
                    #     torch.save(classifier.state_dict(), f'{mweight_fn}/head_fold{i}_{dname}Best-{k}_{expdate}.pt')
                elif k == args.train_obj:
                    patience += 1
            print(log)
            if patience > args.max_patience: break

            # # pbar.set_description(f'Accuracy: {acc:.6f}, F1 Score: {f1:.6f}, Epoch')
            # if f1 >= best_f1:
            #     if f1 > best_f1: 
            #         patience = 0
            #     else:
            #         patience += 1
            #     best_f1 = f1
            #     best_acc = acc
            #     best_prec = prec
            #     best_rec = rec
            #     best_state = model.state_dict()
            #     best_cls_state = classifier.state_dict()
            #     if args.savemodel:
            #         torch.save(model.state_dict(), f'{mweight_fn}/fold{i}_{expdate}.pt')
            # else:
            #     patience += 1
            # if patience > args.max_patience: break
        
        # accuracies.append(best_acc)
        # f1_scores.append(best_f1)
        # prec_scores.append(best_prec)
        # rec_scores.append(best_rec)
        # print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')
        # if args.testname != 'None':
        #     model.load_state_dict(best_state)
        #     classifier.load_state_dict(best_cls_state)
        #     tacc, tprec, trec, tf1 = eval(model, classifier, device, test_loader, hcpatoukb=args.testname in ['hcpa', 'ukb'])
        #     print(f'Testset: Accuracy: {tacc}, F1 Score: {tf1}, Prec: {tprec}, Rec: {trec}')
        #     taccuracies.append(tacc)
        #     tf1_scores.append(tprec)
        #     tprec_scores.append(trec)
        #     trec_scores.append(tf1)
        log = f'Dataset: {args.dataname} [Accuracy, F1 Score, Prec, Rec]:'
        for k in best_acc:
            if k not in accuracies_dict:
                accuracies_dict[k] = []
                f1_scores_dict[k] = []
                prec_scores_dict[k] = []
                rec_scores_dict[k] = []
            accuracies_dict[k].append(best_acc[k])
            f1_scores_dict[k].append(best_f1[k])
            prec_scores_dict[k].append(best_prec[k])
            rec_scores_dict[k].append(best_rec[k])
            log += f'({k}) [{best_acc[k]}, {best_f1[k]}, {best_prec[k]}, {best_rec[k]}], \t'
        print(log)

    # # Calculate mean and standard deviation of evaluation metrics
    for k in accuracies_dict:
        accuracies = accuracies_dict[k]
        f1_scores = f1_scores_dict[k]
        prec_scores = prec_scores_dict[k]
        rec_scores = rec_scores_dict[k]
        auc_score = auc_scores_dict[k]
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = torch.std(torch.tensor(accuracies).float())
        mean_f1_score = sum(f1_scores) / len(f1_scores)
        std_f1_score = torch.std(torch.tensor(f1_scores).float())
        mean_prec_score = sum(prec_scores) / len(prec_scores)
        std_prec_score = torch.std(torch.tensor(prec_scores).float())
        mean_rec_score = sum(rec_scores) / len(rec_scores)
        std_rec_score = torch.std(torch.tensor(rec_scores).float())
        mean_auc_score = sum(auc_score) / len(auc_score)
        std_auc_score = torch.std(torch.tensor(auc_score).float())
        print(f'Dataset: {args.dataname} ({k})')
        print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
        print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
        print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
        print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
        print(f'Mean AUC Score: {mean_auc_score}, Std rec Score: {std_auc_score}')
  

def get_pred_score_pthres(pvalue, pthres, cond):
    if cond == 0:
        ## Smaller p value be more positive
        pred = (pvalue<=pthres).astype(int)
        pred_score = pvalue.copy()
        if len(np.unique(pred_score[pred==1])) > 1:
            pred_score[pred==1] = 1 - (pred_score[pred==1]-pred_score[pred==1].min())/(2*(pred_score[pred==1].max()-pred_score[pred==1].min()))
        else:
            pred_score[pred==1] = 0.5
        if len(np.unique(pred_score[pred==0])) > 1:
            pred_score[pred==0] = 0.5 - (pred_score[pred==0]-pred_score[pred==0].min())/(2*(pred_score[pred==0].max()-pred_score[pred==0].min()))
        else:
            pred_score[pred==0] = 0.5
    elif cond == 1:
        ## Bigger p value be more positive
        pred = (pvalue>pthres).astype(int)
        pred_score = pvalue.copy()
        if len(np.unique(pred_score[pred==1])) > 1:
            pred_score[pred==1] = (pred_score[pred==1]-pred_score[pred==1].min())/(2*(pred_score[pred==1].max()-pred_score[pred==1].min())) + 0.5
        else:
            pred_score[pred==1] = 0.5
        if len(np.unique(pred_score[pred==0])) > 1:
            pred_score[pred==0] = (pred_score[pred==0]-pred_score[pred==0].min())/(2*(pred_score[pred==0].max()-pred_score[pred==0].min()))
        else:
            pred_score[pred==0] = 0.5
    
    return pred, pred_score

def get_pred_score_kmeans(fit_inp, cond, **kargs):
    if len(fit_inp) < 2:
        return np.zeros(len(fit_inp)), np.zeros(len(fit_inp))+0.5
    pred = np.zeros(len(fit_inp))
    kmeans = KMeans(2, random_state=142857, n_init="auto").fit(fit_inp)
    center = np.stack([fit_inp[kmeans.labels_==0].mean(0), fit_inp[kmeans.labels_==1].mean(0)])
    if cond == 0: # small center be positive
        argind = (-center.mean(1)).argsort()
    else: # big center be positive
        argind = (center.mean(1)).argsort()
    pred_score = ((fit_inp - center[argind[1]][None])**2).sum(1)
    pred_score = 1 - ((pred_score-pred_score.min()) / (pred_score.max()-pred_score.min()))
    for predi, kcenteri in enumerate(argind):
        pred[kmeans.labels_==kcenteri] = predi
    return pred, pred_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.linalg as la
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import SpectralClustering
def eval(model, classifier, device, loader, dname=None, train_attn=None, train_y=None, _pthres=0.05, search_train_loader=None):
    model.eval()
    classifier.eval()
    # cumsum_nclass = [0] + list(np.cumsum(loader.dataset.dataset.nclass_list))
    # extract_token_id = []
    # for i in range(len(cumsum_nclass)-1):
    #     extract_token_id.extend(list(np.arange(cumsum_nclass[i]+1, cumsum_nclass[i+1])))

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    all_attn = []
    all_y = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        all_y.append(batch.y.cpu())
        with torch.no_grad():
            feat = model(batch)
            edge_index = batch.edge_index
            batchid = batch.batch
            if len(feat) == 3:  # brainGnn Selected Topk nodes
                feat, edge_index, batchid = feat
            y = classifier(feat, edge_index, batchid, return_cross_attn=True)
        
        attn = y['attn'][:, :, :-3] # B x Nlayer x Ntoken x Nroi
        attn = attn.max(1)[0] # B x Ntoken x Nroi
        all_attn.append(attn)
    all_attn = torch.cat(all_attn)
    all_y = torch.cat(all_y)
    org_y = all_y.clone()
    # all_y[all_y>1] = 1

    if search_train_loader is not None:
        search_attn = []
        search_y = []
        for step, batch in enumerate(search_train_loader):
            batch = batch.to(device)
            search_y.append(batch.y.cpu())
            with torch.no_grad():
                feat = model(batch)
                edge_index = batch.edge_index
                batchid = batch.batch
                if len(feat) == 3:  # brainGnn Selected Topk nodes
                    feat, edge_index, batchid = feat
                y = classifier(feat, edge_index, batchid, return_cross_attn=True)
            
            attn = y['attn'][:, :, :-3] # B x Nlayer x Ntoken x Nroi
            attn = attn.max(1)[0] # B x Ntoken x Nroi
            search_attn.append(attn)
        search_attn = torch.cat(search_attn)
        search_y = torch.cat(search_y)
        org_search_y = search_y.clone()
        print('org_search_y', org_search_y)

    if train_y is not None:
        print(train_attn.shape, train_y.shape)
        train_attn = train_attn.to(device)#[:, extract_token_id] # B x Ntoken x Nroi
        train_y = train_y.to(device)#[:, extract_token_id] # B x Ntoken x Nroi
        print(train_attn.shape, train_y.shape)

        if search_train_loader is not None:
            search_pvalues = np.ones([train_y.shape[1], len(search_attn)])
            for ti in range(train_y.shape[1]): # loop each token
                search2train_corr = torch.corrcoef(torch.cat([train_attn[:, ti], search_attn[:, ti]]))[-len(search_attn):, :-len(search_attn)]
                search2train_corr[search2train_corr.isnan()] = 0
                search2train_corr_p1 = search2train_corr[:, train_y[:, ti]==0]
                search2train_corr_p2 = search2train_corr[:, train_y[:, ti]==1]
                for searchi in range(len(search_attn)): # loop each data
                    search_pvalues[ti, searchi] = ttest_ind(search2train_corr_p1[searchi].cpu().numpy(), search2train_corr_p2[searchi].cpu().numpy()).pvalue
            search_pvalues = search_pvalues[~np.isnan(search_pvalues).all(1)]
            '''
            Grid search for Kmeans condition
            '''
            # search_y_np = search_y[org_search_y.cpu()<=1].cpu().numpy()
            # # fit_inpt = search_pvalues.T[org_search_y.cpu().numpy()<=1]
            # fit_inpt = search2train_corr[org_search_y.cpu().numpy()<=1].detach().cpu().numpy()
            # f1s = np.zeros([2]) # Nsearch x Ntoken
            # accs = np.zeros([2]) # Nsearch x Ntoken
            # aucs = np.zeros([2]) # Nsearch x Ntoken
            # pred, pred_score = get_pred_score_kmeans(fit_inpt, 0)
            # f1_score = precision_recall_fscore_support(search_y_np, pred, average='weighted')[2]
            # fpr, tpr, thresholds = roc_curve(search_y_np, pred_score, pos_label=1)
            # acc = accuracy_score(search_y_np, pred)
            # auc_score = auc(fpr, tpr)
            # f1s[0] = f1_score # f1_score
            # accs[0] = acc # acc
            # aucs[0] = auc_score
            # pred, pred_score = get_pred_score_kmeans(fit_inpt, 1)
            # f1_score = precision_recall_fscore_support(search_y_np, pred, average='weighted')[2]
            # fpr, tpr, thresholds = roc_curve(search_y_np, pred_score, pos_label=1)
            # acc = accuracy_score(search_y_np, pred)
            # auc_score = auc(fpr, tpr)
            # f1s[1] = f1_score # f1_score
            # accs[1] = acc # acc
            # aucs[1] = auc_score
            # print([aucs, f1s, accs])
            # # conditioni = np.lexsort([aucs*-1, f1s*-1, accs*-1])[0]
            # # conditioni = np.argsort(aucs*-1 + f1s*-1 + accs*-1)[0]
            # conditioni = np.argmax(f1s+aucs)
            # # conditioni = np.argmax(accs)

            '''
            Grid search for condition, P threshold and token id
            '''
            pthres = 1
            # pthres_candidates = []
            search_step = 1.1 # log scale 
            search_times = 500
            # search_pvalues = search_pvalues.mean(0, keepdims=True)
            f1s = np.zeros([2, search_times, search_pvalues.shape[0]]) # Nsearch x Ntoken
            accs = np.zeros([2, search_times, search_pvalues.shape[0]]) # Nsearch x Ntoken
            aucs = np.zeros([2, search_times, search_pvalues.shape[0]]) # Nsearch x Ntoken
            searchi = 0
            pbar = tqdm(total=search_times)
            search_y_np = search_y[org_search_y.cpu()<=1].cpu().numpy()
            search_pvalues = search_pvalues[:, org_search_y.cpu().numpy()<=1]
            pthres_candidates = np.zeros([2, search_times, search_pvalues.shape[0]])
            tid_candidates = np.zeros([2, search_times, search_pvalues.shape[0]])
            cond_candidates = np.zeros([2, search_times, search_pvalues.shape[0]])
            while searchi < search_times:
                pbar.update(1)
                # pthres_candidates.append(pthres)
                for ti, pv in enumerate(search_pvalues):
                    pred, pred_score = get_pred_score_pthres(pv, pthres, 0)
                    f1_score = precision_recall_fscore_support(search_y_np, pred, average='weighted')[2]
                    fpr, tpr, thresholds = roc_curve(search_y_np, pred_score, pos_label=1)
                    acc = accuracy_score(search_y_np, pred)
                    auc_score = auc(fpr, tpr)
                    f1s[0, searchi, ti] = f1_score # f1_score
                    accs[0, searchi, ti] = acc # acc
                    aucs[0, searchi, ti] = auc_score
                    pred, pred_score = get_pred_score_pthres(pv, pthres, 1)
                    f1_score = precision_recall_fscore_support(search_y_np, pred, average='weighted')[2]
                    fpr, tpr, thresholds = roc_curve(search_y_np, pred_score, pos_label=1)
                    acc = accuracy_score(search_y_np, pred)
                    f1s[1, searchi, ti] = f1_score # f1_score
                    accs[1, searchi, ti] = acc # acc
                    aucs[1, searchi, ti] = auc_score
                    pthres_candidates[:, searchi, ti] = pthres
                    tid_candidates[:, searchi, ti] = ti
                    cond_candidates[0, searchi, ti] = 0
                    cond_candidates[1, searchi, ti] = 1

                pthres /= search_step
                searchi += 1
            pbar.close()
            fig, axes = plt.subplots(2,3)
            axes = axes.reshape(-1)
            sns.heatmap(f1s[0], ax=axes[0])
            axes[0].set_title('F1 cond 0')
            sns.heatmap(f1s[1], ax=axes[1])
            axes[1].set_title('F1 cond 1')
            sns.heatmap(aucs[0], ax=axes[2])
            axes[2].set_title('AUC cond 0')
            sns.heatmap(aucs[1], ax=axes[3])
            axes[3].set_title('AUC cond 1')
            sns.heatmap(accs[0], ax=axes[4])
            axes[4].set_title('Acc cond 0')
            sns.heatmap(accs[1], ax=axes[5])
            axes[5].set_title('Acc cond 1')
            plt.savefig('tmp.png')
            # sortid = (f1s+aucs).argmax()
            # sortid = np.lexsort([aucs.reshape(-1)*-1, f1s.reshape(-1)*-1, accs.reshape(-1)*-1])[0]
            # sortid = np.lexsort([accs.reshape(-1)*-1, f1s.reshape(-1)*-1])[0]
            # sortid = np.lexsort([aucs.reshape(-1)*-1, accs.reshape(-1)*-1])[0]
            # sortid = np.lexsort([aucs.reshape(-1)*-1, f1s.reshape(-1)*-1])[0]
            sortid = np.argsort(aucs.reshape(-1)*-1 + f1s.reshape(-1)*-1)[0]
            # sortid = np.argsort(f1s.reshape(-1)*-1)[0]
            # sortid = np.argsort(accs.reshape(-1)*-1)[0]
            # sortid = np.lexsort([f1s.reshape(-1)*-1, aucs.reshape(-1)*-1])[0]
            # conditioni, pthresi, searched_ti = np.unravel_index((f1s+aucs).argmax(), f1s.shape)
            # conditioni, pthresi, searched_ti = np.unravel_index(sortid, f1s.shape)
            pthres = pthres_candidates.reshape(-1)[sortid].item()
            searched_ti = int(tid_candidates.reshape(-1)[sortid].item())
            conditioni = cond_candidates.reshape(-1)[sortid].item()
            # pthres = pthres_candidates[pthresi]
            # exit()

        # pvalues = np.ones([train_y.shape[1], 20])
        # sns_data = {'dataID': [], 'tokenID': [], 'group': [], 'corr': []}
        pvalues = np.ones([train_y.shape[1], len(all_attn)])
        for ti in range(train_y.shape[1]):
            test2train_corr = torch.corrcoef(torch.cat([train_attn[:, ti], all_attn[:, ti]]))[-len(all_attn):, :-len(all_attn)]
            test2train_corr[test2train_corr.isnan()] = 0
            test2train_corr_p1 = test2train_corr[:, train_y[:, ti]==0]
            test2train_corr_p2 = test2train_corr[:, train_y[:, ti]==1]
            for testi in range(len(all_attn)):
                pvalues[ti, testi] = ttest_ind(test2train_corr_p1[testi].cpu().numpy(), test2train_corr_p2[testi].cpu().numpy()).pvalue
        ## pvalue distribution plot
        #     for testi in range(20):
        #         pvalue = ttest_ind(test2train_corr_p1[testi].cpu().numpy(), test2train_corr_p2[testi].cpu().numpy()).pvalue
        #         pvalues[ti, testi] = pvalue
        #         sns_data['corr'].extend(test2train_corr_p1[testi].cpu().tolist())
        #         sns_data['dataID'].extend([testi for _ in range(len(test2train_corr_p1[testi]))])
        #         sns_data['tokenID'].extend([ti for _ in range(len(test2train_corr_p1[testi]))])
        #         sns_data['group'].extend([1 for _ in range(len(test2train_corr_p1[testi]))])
        #         sns_data['corr'].extend(test2train_corr_p2[testi].cpu().tolist())
        #         sns_data['dataID'].extend([testi for _ in range(len(test2train_corr_p2[testi]))])
        #         sns_data['tokenID'].extend([ti for _ in range(len(test2train_corr_p2[testi]))])
        #         sns_data['group'].extend([2 for _ in range(len(test2train_corr_p2[testi]))])

        # sns_data = pd.DataFrame(sns_data)
        pvalues = pvalues[~np.isnan(pvalues).all(1)]
        ## pvalues plot
        # plt.figure()
        # plt.imshow(pvalues, cmap='jet', vmax=_pthres)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'figs/zsl/pvalue_{dname}_pthres{_pthres}_{datetime.now()}.png'.replace(' ', '-').replace(':','-'))
        # plt.close()
        ## Acc, F1 plot
        # sns_data = {'F1': [], 'Acc': [], 'Type': [], 'P-thres': []}
        # pthres = 0.95
        # for _ in range(100):
        #     pred = (pvalues<=pthres).astype(int)
        #     # accs = [sum(p==(all_y.cpu().numpy()))/len(all_y) for p in pred]

        #     accs = [accuracy_score(all_y.cpu().numpy(), p) for p in pred]
        #     f1s = [precision_recall_fscore_support(all_y.cpu().numpy(), p, average='weighted')[2] for p in pred]

        #     sns_data['F1'].append(np.max(f1s))
        #     sns_data['Acc'].append(np.max(accs))
        #     sns_data['P-thres'].append(pthres)
        #     sns_data['Type'].append('max')

        #     pred = (pvalues.mean(0)<=pthres).astype(int)
            
        #     acc = accuracy_score(all_y.cpu().numpy(), pred)
        #     f1 = precision_recall_fscore_support(all_y.cpu().numpy(), pred, average='weighted')[2]

        #     # accs = sum(pred==(all_y.cpu().numpy()))/len(all_y)
        #     sns_data['F1'].append(f1)
        #     sns_data['Acc'].append(acc)
        #     sns_data['Type'].append('avg')
        #     sns_data['P-thres'].append(pthres)
        #     pthres /= 2

        # sns_data = pd.DataFrame(sns_data)
        # plt.figure()
        # g = sns.lineplot(data=sns_data, x='P-thres', y='Acc', hue='Type', palette='Blues')
        # g.set_ylabel('Acc', color='blue')
        # g = g.twinx()
        # g = sns.lineplot(data=sns_data, x='P-thres', y='F1', hue='Type', color='b', palette='Reds', ax=g)
        # g.set_ylabel('F1', color='red')
        # plt.tight_layout()
        # plt.xscale('log')
        # plt.savefig(f'figs/zsl/pthresVSacc_{dname}_{datetime.now()}.png'.replace(' ', '-').replace(':','-'))
        # plt.close()
        ## Correlation plot
        # g = sns.catplot(data=sns_data, x='group', y='corr', row='tokenID', col='dataID', kind='box')
        # # g.map(plt.title, 'pvalue', color='black', lw=0)
        # # g.set_titles('{col_name}')
        # plt.tight_layout()
        # plt.savefig(f'figs/zsl/corr_{dname}_{datetime.now()}.png'.replace(' ', '-').replace(':','-'))
        # exit()

        ## threshold
        all_y_np = all_y[org_y.cpu()<=1].cpu().numpy()
        # pvalue = pvalues[:, org_y.cpu().numpy()<=1].mean(0)
        # pvalue = pvalues.mean(0, keepdims=True)[searched_ti, org_y.cpu().numpy()<=1]
        if search_train_loader is not None:
            pvalue = pvalues[searched_ti, org_y.cpu().numpy()<=1]
            pred, pred_score = get_pred_score_pthres(pvalue, pthres, conditioni)
        else:
            pvalue = pvalues[:, org_y.cpu().numpy()<=1].mean(0)
            pred, pred_score = get_pred_score_pthres(pvalue, _pthres, 0)
        acc = accuracy_score(all_y_np, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(all_y_np, pred, average='weighted')
        fpr, tpr, thresholds = roc_curve(all_y_np, pred_score, pos_label=1)
        auc_area = auc(fpr, tpr)

        ## force 2-class
        # kmeans = KMeans(2, random_state=0, n_init="auto").fit(pvalues.T)
        # argind = (-kmeans.cluster_centers_.mean(1)).argsort()
        # pred = np.zeros_like(all_y.cpu().numpy())
        # for predi, kcenteri in enumerate(argind):
        #     pred[kmeans.labels_==kcenteri] = predi      
        # acc = accuracy_score(all_y.cpu().numpy(), pred)
        # prec, rec, f1, _ = precision_recall_fscore_support(all_y.cpu().numpy(), pred, average='weighted')

        ## only 0 and 1 class
        # fit_inpt = pvalues.T[org_y.cpu().numpy()<=1]
        # # fit_inpt = test2train_corr[org_y.cpu().numpy()<=1].detach().cpu().numpy()
        # all_y_np = all_y[org_y.cpu()<=1].cpu().numpy()
        # if search_train_loader is not None:
        #     pred, pred_score = get_pred_score_kmeans(fit_inpt, conditioni)
        # else:
        #     pred, pred_score = get_pred_score_kmeans(fit_inpt, 0)
        # # print(pred_score, pred)
        # acc = accuracy_score(all_y_np, pred)
        # prec, rec, f1, _ = precision_recall_fscore_support(all_y_np, pred, average='weighted')
        # fpr, tpr, thresholds = roc_curve(all_y_np, pred_score, pos_label=1)
        # auc_area = auc(fpr, tpr)

        ## 4-class
        # kmeans = KMeans(len(org_y.unique()), random_state=0, n_init="auto").fit(pvalues.T)
        # argind = (-np.median(kmeans.cluster_centers_, 1)).argsort()
        # pred = np.zeros_like(org_y.cpu().numpy())
        # for predi, kcenteri in enumerate(argind):
        #     pred[kmeans.labels_==kcenteri] = predi
        # acc = accuracy_score(org_y.cpu().numpy(), pred)
        # prec, rec, f1, _ = precision_recall_fscore_support(org_y.cpu().numpy(), pred, average='weighted')
        print(list(all_y_np), list(pred))
        return acc, f1, prec, rec, auc_area

        
    # corr = torch.stack([torch.corrcoef(all_attn[:, i]) for i in range(all_attn.shape[1])]).cpu().numpy()
    # corr[:, np.arange(corr.shape[1]), np.arange(corr.shape[1])] = 0
    # fig, axes = plt.subplots(3, 7, sharex=True, sharey=True)
    # axes = axes.reshape(-1)
    # for i in range(len(corr)):
    #     ind = np.lexsort(np.abs(corr[i]))
    #     one_corr = np.abs(corr[i][ind, :][:, ind])
    #     axes[i].imshow(one_corr, cmap='coolwarm', interpolation='nearest')
    # plt.tight_layout()
    # plt.savefig(f'figs/zsl/corr_{dname}_{datetime.now()}.png'.replace(' ', '-').replace(':','-'))
    # plt.close()
    return all_attn.cpu(), all_y
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

def graph_laplacian(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix

def spectral_partitioning(laplacian_matrix):
    eigenvalues, eigenvectors = la.eig(laplacian_matrix)
  
    # Sort eigenvalues and eigenvectors in ascending order of eigenvalues
    index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    fiedler_vector = eigenvectors[:, 1]  # Second smallest eigenvalue

    partition1 = np.where(fiedler_vector >= 0)[0]
    partition2 = np.where(fiedler_vector < 0)[0]
    return partition1, partition2

def braingnn_nodesz(node_sz, ratio):
    if node_sz != 333:
        return math.ceil(node_sz*ratio*ratio)
    else:
        return 31

if __name__ == '__main__': main()