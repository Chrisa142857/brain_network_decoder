from scipy import stats
import numpy as np

with open('lcmbig_ttest.log', 'r') as file:
    lines = file.read().split('\n')

rvs = {'lcm': {}, 'other': {}}
for l in lines:
    if l == '': continue
    if '# LCM-big*' in l:
        k = 'lcm'
    elif l.startswith('#'):
        k = 'other'
    # print(k)
    if l.startswith('#'): continue
    dn = l.split(' [')[0]
    l = l.replace(dn, '').replace(' ', '').replace('[', '').replace(']', '')
    # print(l)
    rvs[k][dn] = [float(item) for item in l.split(',')]
    rvs[k][dn] = [i*100 if i <= 1 else i for i in rvs[k][dn]]
# print(rvs)
# exit()
dn_rvs = {}
for k in rvs:
    for dn in rvs[k]:
        if dn not in dn_rvs: dn_rvs[dn] = {}
        dn_rvs[dn][k] = np.sort(rvs[k][dn])
        # dn_rvs[dn][k] = rvs[k][dn]
# print(dn_rvs)
# exit()
for dn in dn_rvs:
    print(len(dn_rvs[dn]['lcm']), len(dn_rvs[dn]['other']))
    num = min(len(dn_rvs[dn]['lcm']), len(dn_rvs[dn]['other']))
    res = stats.ttest_rel(dn_rvs[dn]['lcm'][:num], dn_rvs[dn]['other'][:num])
    # res = stats.ttest_ind(dn_rvs[dn]['lcm'], dn_rvs[dn]['other'])
    print(dn, res)