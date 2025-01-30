
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def fig1(savetag='', dtype='attrFC', backbone='none', multiClass=False):
    skip_dn = ['oasis']
    # decoder = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': []}
    # mlp = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': []}

    # data = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': [], 'model': []}
    data = {'layer #': [], 'model': [], 'score': [], 'metric': [], 'dataset': []}
    avg_loss = {'layer #': [], 'model': [], 'dataset': [], 'loss': []}
    for logf in os.listdir(f'logs'):
        if not logf.endswith('.log'): continue
        # if dname not in logf: continue
        
        with open(f'logs/{logf}', 'r') as f:
            lines = f.read().split('\n')
        if len(lines) <= 5: continue
        if 'Mean' not in lines[-5]: continue
        bb = logf.split('_')[0]
        if bb != backbone: continue
        cls = logf.split('_')[2]
        
        if multiClass:
            if 'Mix' not in cls: continue
            layern = int(cls.replace('8Mix' if 'decoder' in cls else 'LMix','').replace('decoder' if 'decoder' in cls else 'mlp', ''))
        else:
            if 'Mix' in cls: continue
            layern = int(cls.replace('decoder' if 'decoder' in cls else 'mlp', ''))

        dn = logf.split('_')[1]
        if dn in skip_dn: continue
        dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
        dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
        if dt != dtype: continue
        # acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
        # acc_std = float(lines[0].split('Accuracy: ')[2])
        # f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
        # f1_std = float(lines[1].split('F1 Score: ')[2])
        res_lines = [l for l in lines if l.startswith('Accuracy') and 'Epoch' not in l]
        res = {}
        print(logf)
        for l in res_lines:
            for item in l.split(','):
                k, v = item.split(': ')
                v = float(v) * 100
                if k not in res: res[k] = []
                res[k].append(v)

        for k in res:
            for item in res[k]:
                data['layer #'].append(layern)
                data['model'].append('decoder' if 'decoder' in cls else 'mlp')
                data['score'].append(item)
                data['metric'].append(k)
                data['dataset'].append(dn)

        loss_lines = [l for l in lines if 'Train loss' in l or l.startswith('Fold')]
        loss = []
        one_fold_loss = None
        for l in loss_lines:
            if l.startswith('Fold'):
                if one_fold_loss is not None: loss.append(np.mean(one_fold_loss))
                one_fold_loss = []
                continue
            one_loss = float(l.split('Train loss')[1].split(',')[0].split(': ')[1])
            one_fold_loss.append(one_loss)
        for lossi in loss:
            avg_loss['layer #'].append(layern)
            avg_loss['model'].append('decoder' if 'decoder' in cls else 'mlp')
            avg_loss['dataset'].append(dn)
            avg_loss['loss'].append(lossi)
            
    data = pd.DataFrame(data)
    avg_loss = pd.DataFrame(avg_loss)
    plt.figure()
    sns.relplot(data=data, x='layer #', y='score', 
                row='model', col='dataset', hue="metric", style="metric", 
                kind="line", height=3, aspect=.5,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_lineplot{savetag}.png')
    plt.savefig(f'figs/fig1_lineplot{savetag}.svg')
        
    plt.figure()
    sns.lmplot(data=data, x='layer #', y='score', x_jitter=.15, scatter=False,
                row='model', col='dataset', hue="metric", 
                height=3, aspect=.5, order=2,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_regplot{savetag}.png')
    plt.savefig(f'figs/fig1_regplot{savetag}.svg')
        
        
    plt.figure()
    sns.relplot(data=avg_loss, x='layer #', y='loss', 
                col='dataset', hue="model", style="model", 
                kind="line", height=3, aspect=.5,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_loss_lineplot{savetag}.png')
    plt.savefig(f'figs/fig1_loss_lineplot{savetag}.svg')
    
    plt.figure()
    sns.lmplot(data=avg_loss, x='layer #', y='loss', x_jitter=.15, scatter=False,
                col='dataset', hue="model", 
                height=3, aspect=.5, order=2,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_loss_regplot{savetag}.png')
    plt.savefig(f'figs/fig1_loss_regplot{savetag}.svg')