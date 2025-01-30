from datasets import dataloader_generator
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
    'sex': nn.CrossEntropyLoss(),
    'age': nn.MSELoss(), 
}
LOSS_W = {
    'y': 1,
    'sex': 1,
    'age': 1e-4,
}

def main():
    parser = argparse.ArgumentParser(description='NeuroDetour')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default = 200)
    parser.add_argument('--models', type=str, default = 'bnt')
    parser.add_argument('--classifier', type=str, default = 'mlp')
    parser.add_argument('--max_patience', type=int, default = 50)
    parser.add_argument('--hiddim', type=int, default = 2048)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--atlas', type=str, default = 'AAL_116')
    parser.add_argument('--dataname', type=str, default = 'ppmi')
    parser.add_argument('--testname', type=str, default = 'None')
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
    parser.add_argument('--decoder_layer', type=int, default = 8)
    parser.add_argument('--train_obj', type=str, default = 'y')

    args = parser.parse_args()
    print(args)
    expdate = str(datetime.now())
    expdate = expdate.replace(':','-').replace(' ', '-').replace('.', '-')
    device = args.device
    hiddim = args.hiddim
    nclass = DATA_CLASS_N[args.dataname]
    dataset = None
    # Initialize lists to store evaluation metrics
    accuracies_dict = {}
    f1_scores_dict = {}
    prec_scores_dict = {}
    rec_scores_dict = {}
    taccuracies = []
    tf1_scores = []
    tprec_scores = []
    trec_scores = []
    node_sz = ATLAS_ROI_N[args.atlas]
    # if args.models != 'neurodetour':
    transform = None
    dek, pek = 0, 0
    if args.node_attr != 'BOLD':
        input_dim = node_sz
    else:
        input_dim = args.bold_winsize
    transform = DATA_TRANSFORM[args.models]
    testset = args.testname
    if args.savemodel:
        mweight_fn = f'model_weights/{args.models}_{args.atlas}_boldwin{args.bold_winsize}_{args.adj_type}{args.node_attr}'
        os.makedirs(mweight_fn, exist_ok=True)
    for i in range(args.cv_fold_n):
        dataloaders = dataloader_generator(batch_size=args.batch_size, nfold=i, dataset=dataset, total_fold=args.cv_fold_n,
                                                                 node_attr=args.node_attr, adj_type=args.adj_type, transform=transform, dname=args.dataname, testset=testset,
                                                                 fc_winsize=args.bold_winsize, atlas_name=args.atlas, fc_th=args.fc_th, sc_th=args.sc_th)
        if args.only_dataload: exit()
        if len(dataloaders) == 3:
            train_loader, val_loader, dataset = dataloaders
        else:
            train_loader, val_loader, dataset, test_loader, testset = dataloaders
        model = MODEL_BANK[args.models](node_sz=node_sz, out_channel=hiddim, in_channel=input_dim, batch_size=args.batch_size, device=device, nlayer=args.nlayer, heads=args.nhead).to(device)
        # print(sum([p.numel() for p in model.parameters()]))
        # exit()
        if not args.decoder:
            classifier = Classifier(CLASSIFIER_BANK[args.classifier], hiddim, nlayer=args.decoder_layer, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), aggr=args.classifier_aggr).to(device)
        else:
            classifier = BNDecoder(hiddim, nclass=nclass, node_sz=node_sz if args.models!='braingnn' else braingnn_nodesz(node_sz, model.ratio), nlayer=args.decoder_layer, head_num=8, return_intermediate=False).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.decay) 
        # print(optimizer)
        # best_f1 = 0
        patience = 0
        best_f1 = {}
        best_acc = {}
        best_prec = {}
        best_rec = {}
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            print(datetime.now(), 'train start')
            train(model, classifier, device, train_loader, optimizer)
            print(datetime.now(), 'train done, test start')
            # acc, prec, rec, f1 = eval(model, classifier, device, val_loader)
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
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = torch.std(torch.tensor(accuracies).float())
        mean_f1_score = sum(f1_scores) / len(f1_scores)
        std_f1_score = torch.std(torch.tensor(f1_scores).float())
        mean_prec_score = sum(prec_scores) / len(prec_scores)
        std_prec_score = torch.std(torch.tensor(prec_scores).float())
        mean_rec_score = sum(rec_scores) / len(rec_scores)
        std_rec_score = torch.std(torch.tensor(rec_scores).float())
        print(f'Dataset: {args.dataname} ({k})')
        print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
        print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
        print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
        print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')

    # mean_accuracy = sum(accuracies) / len(accuracies)
    # std_accuracy = torch.std(torch.tensor(accuracies))
    # mean_f1_score = sum(f1_scores) / len(f1_scores)
    # std_f1_score = torch.std(torch.tensor(f1_scores))
    # mean_prec_score = sum(prec_scores) / len(prec_scores)
    # std_prec_score = torch.std(torch.tensor(prec_scores))
    # mean_rec_score = sum(rec_scores) / len(rec_scores)
    # std_rec_score = torch.std(torch.tensor(rec_scores))

    # print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    # print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    # print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    # print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')

    # if args.testname != 'None':
    #     mean_accuracy = sum(taccuracies) / len(taccuracies)
    #     std_accuracy = torch.std(torch.tensor(taccuracies))
    #     mean_f1_score = sum(tf1_scores) / len(tf1_scores)
    #     std_f1_score = torch.std(torch.tensor(tf1_scores))
    #     mean_prec_score = sum(tprec_scores) / len(tprec_scores)
    #     std_prec_score = torch.std(torch.tensor(tprec_scores))
    #     mean_rec_score = sum(trec_scores) / len(trec_scores)
    #     std_rec_score = torch.std(torch.tensor(trec_scores))
    #     print(f'Test set: {args.testname}')
    #     print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    #     print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    #     print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    #     print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
        
def train(model, classifier, device, loader, optimizer):
    model.train()
    classifier.train()
    losses = []
    y_true_dict = {k: [] for k in LOSS_FUNCS}
    y_scores_dict = {k: [] for k in LOSS_FUNCS}
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
            loss += LOSS_W[k]*LOSS_FUNCS[k](y[k][batch[k] != -1], batch[k][batch[k] != -1])
        if hasattr(model, 'loss'):
            loss = loss + model.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        for k in LOSS_FUNCS:
            y_true_dict[k].append(batch[k][batch[k] != -1])
            y_scores_dict[k].append(y[k][batch[k] != -1].detach().cpu())
    
    logs = [f'Train loss: {np.mean(losses):.6f}']
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu()
        if k != 'age':
            y_true = y_true.numpy()
            y_scores = y_scores.numpy().argmax(1)
            acc = accuracy_score(y_true, y_scores)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
            logs.append(f'{k}-Accuracy: {acc:.6f}')
        else:
            logs.append(f'{k}-MSE: {torch.nn.functional.mse_loss(y_scores, y_true):.6f}')
    
    print(', '.join(logs))

def eval(model, classifier, device, loader, hcpatoukb=False):
    model.eval()
    classifier.eval()
    # y_true = []
    # y_scores = []
    y_true_dict = {k: [] for k in LOSS_FUNCS}
    y_scores_dict = {k: [] for k in LOSS_FUNCS}

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
            # pred = classifier(feat, edge_index, batchid)
            # pred = pred['y']

    #     y_true.append(batch.y)
    #     y_scores.append(pred.detach().cpu())

    # y_true = torch.cat(y_true, dim = 0).detach().cpu().numpy()
    # y_scores = torch.cat(y_scores, dim = 0).numpy().argmax(1)
    # if hcpatoukb:
    #     y_scores[y_scores>1] = 1
    #     y_true[y_true>1] = 1
    # acc = accuracy_score(y_true, y_scores)
    # prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    # return acc, prec, rec, f1

        for k in LOSS_FUNCS:
            y_true_dict[k].append(batch[k][batch[k] != -1])
            y_scores_dict[k].append(y[k][batch[k] != -1].detach().cpu())

    scores = {}    
    for k in y_true_dict:
        y_true = torch.cat(y_true_dict[k], dim = 0).detach().cpu()
        y_scores = torch.cat(y_scores_dict[k], dim = 0).detach().cpu()
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