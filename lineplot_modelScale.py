import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

model_param = {
    'MLP': {4:151793788,8:302813308,12:453832828,16:604852348,20:755871868,24:906891388,28:1057910908,32:1208930428},
    'mlp': {4:15179378.8,8:30281330.8,12:45383282.8,16:60485234.8,20:75587186.8,24:90689138.8,28:105791090.8,32:120893042.8},
    'decoder': {4:147194883,8:294121475,12:441048067,16:587974659,20:734901251,24:881827843,28:1028754435,32:1175681027}
}


valid_dn = ['taowu', 'adni', 'ppmi', 'abide', 'neurocon'] # , 'hcpa', 'hcpya'
# data = {'backbone': [], 'classifier': [], 'dataset': [], 'data type': [], 'acc': [], 'f1': []}
data = {'type': [], 'param': [], 'dataset': [], 'score': [], 'metric': [], 'std': []}
logtag = 'decoder_vs_mlp'
for logf in os.listdir(f'logs'):
    if not logf.endswith('.log'): continue
    if 'Mix' in logf: continue
    with open(f'logs/{logf}', 'r') as f:
        lines = f.read().split('\n')
    if len(lines) <= 5: continue
    if '(age)' in lines[-6]:
        lines = lines[-15:-11]
    else:
        lines = lines[-5:]
    
    if 'Mean' not in lines[0]: continue
    bb = logf.split('_')[0]
    if bb != 'none': continue
    cls = logf.split('_')[2]
    # if 'decoder' not in cls: continue
    if 'MLP' not in cls and 'mlp' not in cls: continue
    dn = logf.split('_')[1]
    if dn not in valid_dn: continue
    dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
    dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
    if dt == 'attrBOLD': continue
    acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
    acc_std = float(lines[0].split('Accuracy: ')[2])
    f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
    f1_std = float(lines[1].split('F1 Score: ')[2])
    # data['backbone'].append(bb)
    if 'Mix' not in cls: 
        layern = int(cls.upper().replace('decoder' if 'decoder' in cls else 'MLP', ''))
        # cls = 'w/o S&A.'
    else:
        layern = int(cls.upper().replace('8Mix' if 'decoder' in cls else 'LMix','').replace('decoder' if 'decoder' in cls else 'MLP', ''))
        # cls = 'w/ S&A'

    # cls = 'MLP'
    cls = cls[:3]
    # data['type'].append(cls)
    data['type'].append(cls)
    # data['param'].append(model_param[cls][layern])
    data['param'].append(model_param[cls][layern])
    # data['dataset'].append(dn)
    data['dataset'].append(dn)
    # data['score'].append(acc_avg*100)
    data['score'].append(f1_avg*100)
    # data['metric'].append('Acc')
    data['metric'].append('F1')
    # data['std'].append(acc_std*100)
    data['std'].append(f1_std*100)

data = pd.DataFrame(data)
print(data)
# exit()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})
plt.figure()
g = sns.lmplot(data=data, x='param', y='score', hue='metric', col='dataset', row='type', height=3, sharex=False, sharey=False)
# g.map_dataframe(errplot, "param", "score", "std")
# fig, ax = plt.subplots(2, 4, sharex=True)
# ax = g.axes.reshape(-1)
# for i, d in enumerate(data['dataset'].unique()):
#     one_data = data[data['dataset']==d]
#     one_one_data = one_data[one_data['metric']=='Acc']
#     ax[i].errorbar(x=one_one_data['param'], y=one_one_data['score'], yerr=one_one_data['std'], fmt='none', capsize=5, ecolor='blue')
#     one_one_data = one_data[one_data['metric']=='F1']
#     ax[i].errorbar(x=one_one_data['param'], y=one_one_data['score'], yerr=one_one_data['std'], fmt='none', capsize=5, ecolor='orange')
import scipy as sp

ann_count = {}

for d in data['dataset'].unique():
    for t in data['type'].unique():
        ann_count[f'{d}-{t}'] = 0

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['param'], data['score'])
    d, t = data['dataset'].unique()[0], data['type'].unique()[0]
    ann_count[f'{d}-{t}'] += 1
    if ann_count[f'{d}-{t}'] == 1:
        x, y = .05, .8
    else:
        x, y = .05, .7
    ax = plt.gca()
    ax.text(x, y, 'r={:.2f}'.format(r),
            transform=ax.transAxes)
    
g.map_dataframe(annotate)

plt.tight_layout()
plt.savefig(f'figs/exp_MLP_regplot_modelScaleVSacc.png')
plt.savefig(f'figs/exp_MLP_regplot_modelScaleVSacc.svg')