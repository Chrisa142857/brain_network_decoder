import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
valid_dn = ['taowu', 'adni', 'ppmi', 'ppmi2cls', 'abide', 'neurocon', 'hcpa', 'hcpya', 'sz-diana', 'sz-diana-hard']
# data = {'backbone': [], 'classifier': [], 'data': [], 'data type': [], 'acc': [], 'f1': []}
# data = {'backbone': [], 'classifier': [], 'data': [], 'acc': [], 'f1': []}
data = {'backbone': [], 'classifier': [], 'data': [], 'acc_avg': [], 'f1_avg': [], 'acc_std': [], 'f1_std': []}
logtag = 'logs3'
for logf in os.listdir(f'logs3'):
    if not logf.endswith('.log'): continue
    if 'mlp1Mix' in logf: continue
    with open(f'logs3/{logf}', 'r') as f:
        lines = f.read().split('\n')[-5:]
    if 'Mean' not in lines[0]: continue
    bb = logf.split('_')[0]
    if bb == 'gcn': continue
    cls = logf.split('_')[2]
    dn = logf.split('_')[1]
    if dn not in valid_dn: continue
    dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
    dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
    if dt == 'attrBOLD': continue
    if len(lines[0].split('Accuracy: ')) < 3: continue
    acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
    acc_std = float(lines[0].split('Accuracy: ')[2])
    f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
    f1_std = float(lines[1].split('F1 Score: ')[2])
    data['backbone'].append(bb)
    data['classifier'].append(cls)
    data['data'].append(dn)
    # data['data type'].append(dt)
    # data['acc'].append(f"{(acc_avg*100):.5f}+-{(acc_std*100):.5f}")
    # data['f1'].append(f"{(f1_avg*100):.5f}+-{(f1_std*100):.5f}")

    data['acc_avg'].append(acc_avg*100)
    data['f1_avg'].append(f1_avg*100)
    data['acc_std'].append(acc_std*100)
    data['f1_std'].append(f1_std*100)

data = pd.DataFrame(data)
# one = data[data['backbone']=='none'][data['classifier']=='decoder32']
# x = one[one['data']=='adni']
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# ad = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# x = one[one.data.isin(['ppmi', 'taowu', 'neurocon'])]
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# pd = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# x = one[one['data']=='abide']
# acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
# f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
# aut = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#         f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
# print(' & '.join(ad + pd + aut))
# # exit()
# # interested_bb = ['braingnn', 'bnt', 'bolt', 'graphormer', 'nagphormer', 'neurodetour']
# interested_bb = ['braingnn', 'bnt', 'bolt', 'graphormer', 'nagphormer', 'neurodetour']
# for bb in interested_bb:
#     one = data[data['backbone']==bb]
#     print(bb)
#     x = one[one['data']=='adni']
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     ad = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     x = one[one.data.isin(['ppmi', 'taowu', 'neurocon'])]
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     pd = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     x = one[one['data']=='abide']
#     acc = [x['acc_avg'].mean(), x['acc_std'].mean()]
#     f1 = [x['f1_avg'].mean(), x['f1_std'].mean()]
#     aut = [f'{float(acc[0]):.2f}$_'+ '{\pm'+f'{float(acc[1]):.2f}'+'}$', 
#           f'{float(f1[0]):.2f}$_'+ '{\pm'+f'{float(f1[1]):.2f}'+'}$']
#     print(' & '.join(ad + pd + aut))

data = data.sort_values('classifier')
data = data.sort_values('backbone')
# data = data.sort_values('data type')
results = '\n'
for unid in data['data'].unique():
    df = data[data['data']==unid]
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.drop("data",axis=1,inplace=True)
    results += f'\n\n## Data: {unid} \n\n'
    results += df.to_markdown()
    results += '\n'

with open(f'{logtag}_in_markdown.md', 'w') as f:
    f.write(results)
exit()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fig1(savetag='', dtype='attrFC', backbone='none', multiClass=False):

    model_param = {
        'mlp': {4:151793788,8:302813308,12:453832828,16:604852348,20:755871868,24:906891388,28:1057910908,32:1208930428},
        'decoder': {4:147194883,8:294121475,12:441048067,16:587974659,20:734901251,24:881827843,28:1028754435,32:1175681027}
    }
    skip_dn = ['oasis']
    # decoder = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': []}
    # mlp = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': []}

    # data = {'layer #': [], 'acc': [], 'f1': [], 'acc_std': [], 'f1_std': [], 'model': []}
    data = {'layer #': [], 'param #': [], 'model': [], 'score': [], 'metric': [], 'dataset': []}
    avg_loss = {'layer #': [], 'param #': [], 'model': [], 'dataset': [], 'loss': []}
    for logf in os.listdir(f'logs'):
        if not logf.endswith('.log'): continue
        if '-' in logf: continue
        if 'MLP' in logf: continue
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
                data['param #'].append(model_param[data['model'][-1]][layern])
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
            avg_loss['param #'].append(model_param[avg_loss['model'][-1]][layern])
            avg_loss['dataset'].append(dn)
            avg_loss['loss'].append(lossi)
            
    data = pd.DataFrame(data)
    avg_loss = pd.DataFrame(avg_loss)
    plt.figure()
    sns.relplot(data=data, x='param #', y='score', 
                row='model', col='dataset', hue="metric", style="metric", 
                kind="line", height=3, aspect=.5,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_lineplot{savetag}.png')
    plt.savefig(f'figs/fig1_lineplot{savetag}.svg')
        
    plt.figure()
    sns.lmplot(data=data, x='param #', y='score', x_jitter=.15, scatter=False,
                row='model', col='dataset', hue="metric", 
                height=3, aspect=.5, order=2,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_regplot{savetag}.png')
    plt.savefig(f'figs/fig1_regplot{savetag}.svg')
        
        
    plt.figure()
    sns.relplot(data=avg_loss, x='param #', y='loss', 
                col='dataset', hue="model", style="model", 
                kind="line", height=3, aspect=.5,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_loss_lineplot{savetag}.png')
    plt.savefig(f'figs/fig1_loss_lineplot{savetag}.svg')
    
    plt.figure()
    sns.lmplot(data=avg_loss, x='param #', y='loss', x_jitter=.15, scatter=False,
                col='dataset', hue="model", 
                height=3, aspect=.5, order=2,
                facet_kws={'sharey': False, 'sharex': False})
    plt.tight_layout()
    plt.savefig(f'figs/fig1_loss_regplot{savetag}.png')
    plt.savefig(f'figs/fig1_loss_regplot{savetag}.svg')
        
fig1(savetag='_multiCls', multiClass=True)        
fig1(savetag='_singleY', multiClass=False)