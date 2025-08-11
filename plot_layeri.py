import numpy as np
import pandas as pd
import seaborn as sns

labels = []
layeris = []


def readdata(dname = 'ppmi'):
    data = np.load(f'{dname}_lcm_layeri.npy', allow_pickle=True).item()
    gt = data['gt']
    layeri = data['layeri_selected']
    # print(data)

    if dname != 'adni':
        layeri = layeri[gt!=1]
        gt = gt[gt!=1]

    return gt, layeri

classname = ['CN', 'Parkinson\'s', 'Prodromal', 'SWEDD']
gt, layeri = readdata('ppmi')
layeris.extend(list(layeri+1))
labels += [classname[i-1] for i in gt]

classname = ['CN', 'Alzheimer\'s']
gt, layeri = readdata('adni')
layeris.extend(list(layeri+1))
labels += [classname[i-1] for i in gt]

classname = ['CN', 'Autism']
gt, layeri = readdata('abide')
layeris.extend(list(layeri+1))
labels += [classname[i-1] for i in gt]

# classname = ['task-SOCIAL', 'task-GAMBLING', 'task-EMOTION', 'task-WM', 'task-MOTOR', 'task-RELATIONAL', 'task-LANGUAGE']
# gt, layeri = readdata('hcpya')
# layeris.extend(list(layeri+1))
# labels += [classname[i-1] for i in gt]

# classname = ['task-REST', 'task-CARIT', 'task-FACENAME', 'task-VISMOTOR']
# gt, layeri = readdata('hcpa')
# layeris.extend(list(layeri+1))
# labels += [classname[i-1] for i in gt]

# ind = [i for i in range(len(labels)) if labels[i] != 'CN']
ind = np.arange(len(labels))
data = pd.DataFrame({
    'Label': np.array(labels)[ind],
    'Layer Index': np.array(layeris)[ind]
})

import matplotlib.pyplot as plt
plt.figure(figsize=[10,2.5])
sns.boxplot(data=data, y='Label', x='Layer Index', color='lightgrey', width=.5)
sns.stripplot(data=data, y='Label', x='Layer Index', color="black", alpha=0.7, jitter=True)
plt.savefig('layeri_dist.png')
plt.savefig('layeri_dist.svg')

exit()

import plotext as plt
import random

dist_table = {
    'Layer ID': [],
    'Freq': [],
    'Diagnose': []
}
dist_table = [['Diag',]]


label = np.array(labels)[ind]
x = np.array(layeris)[ind]
for l in np.unique(label):
    data = x[label==l]
    bins = 8
    hist, bin_edge = np.histogram(data, bins=[i for i in range(0, 32, bins)]+[32])
    
    dist_table.append([l])
    for i in range(len(hist)):
        dist_table[-1].append(f'{hist[i]/len(data):.1f}')
        # dist_table['Freq'].append(hist[i])
        if len(dist_table) == 2: dist_table[0].append(f'{bin_edge[i]+1} ~ {bin_edge[i+1]}')
        # dist_table['Layer ID'].append(f'{bin_edge[i]+1} ~ {bin_edge[i+1]}')
        # dist_table['Diagnose'].append(l)

    # plt.hist(data, bins)
    # plt.xlim(1, 32)
    # plt.xticks([i+1 for i in range(32)], [i+1 for i in range(32)])
    # plt.show()
    # plt.save_fig(f'/ram/USERS/ziquanw/brain_network_decoder/layeri_dist_{l}.txt', keep_colors=False)
    # plt.clear_figure()
print([len(l) for l in dist_table])
dist_table = pd.DataFrame(dist_table)
print(dist_table.to_markdown(index=False))