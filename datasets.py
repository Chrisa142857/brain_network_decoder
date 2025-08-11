import os, torch, difflib
from scipy.io import loadmat
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
# import networkx as nx
from tqdm import tqdm, trange
# from statannotations.Annotator import Annotator
# from scipy.stats import ttest_rel, ttest_ind
from torch_geometric.data import Data
# from torch_geometric.utils import remove_self_loops, add_self_loops
# from torch.nn.utils.rnn import pad_sequence


ATLAS_FACTORY = ['AAL_116', 'Aicha_384', 'Gordon_333', 'Brainnetome_264', 'Shaefer_100', 'Shaefer_200', 'Shaefer_400', 'D_160']
BOLD_FORMAT = ['.csv', '.csv', '.tsv', '.csv', '.tsv', '.tsv', '.tsv', '.txt']
DATAROOT = {
    'adni': '../detour_hcp/data',
    'oasis': '../detour_hcp/data',
    'hcpa': '../Lab_DATA',
    'ukb': '../data',
    'hcpya': '../data',
    'ppmi': '../benchmark_fmri/data/PPMI',
    'abide': '../benchmark_fmri/data/ABIDE',
    'neurocon': '../Lab_DATA/All_Dataset/neurocon/neurocon',
    'taowu': '../Lab_DATA/All_Dataset/taowu/taowu',
}
DATANAME = {
    'adni': 'ADNI_BOLD_SC',
    'oasis': 'OASIS_BOLD_SC',
    'hcpa': 'HCP-A-SC_FC',
    'ukb': 'UKB-SC-FC',
    'hcpya': 'HCP-YA-SC_FC',
}
LABEL_NAME_P = {
    'adni': -1, 'oasis': -1, 
    'hcpa': 1, 'hcpya': 1, 
    'ukb': 2,
}

LABEL_REMAP = {
    'adni': {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'CN', 'LMCI': 'AD', 'AD': 'AD'},
    'oasis': {'CN': 'CN', 'AD': 'AD'},
}
DATA_DEFAULT_TYPE = {
    'adni': 'task-REST',
    'hcpa': 'CN',
    'hcpya': 'CN',
    'ppmi': 'task-REST',
    'abide': 'task-REST',
    'neurocon': 'task-REST',
    'taowu': 'task-REST',
}
DISEASE_DATA = ['adni', 'ppmi', 'abide', 'neurocon', 'taowu']
class NeuroNetworkDataset(Dataset):

    def __init__(self, atlas_name='AAL_116',
                 dname='hcpa',
                node_attr = 'SC', adj_type = 'FC',
                transform = None,
                fc_winsize = 500,
                fc_winoverlap = 0,
                fc_th = 0.5,
                sc_th = 0.1) -> None:
        default_fc_th = 0.5
        default_sc_th = 0.1
        data_root = DATAROOT[dname]
        data_name = DATANAME[dname]
        self.transform = transform
        self.data_root = f"{data_root}/{data_name}"
        self.fc_winsize = fc_winsize
        self.fc_th = fc_th
        self.sc_th = sc_th
        self.dname = dname
        subn_p = 0
        subtask_p = LABEL_NAME_P[dname]
        # subdir_p = 2
        # bold_format = BOLD_FORMAT[ATLAS_FACTORY.index(atlas_name)]
        # fc_format = '.csv'
        assert atlas_name in ATLAS_FACTORY, atlas_name
        bold_root = f'{self.data_root}/{atlas_name}/BOLD'
        fc_root = f'{self.data_root}/{atlas_name}/FC'
        sc_root = f'{self.data_root}/ALL_SC'
        atlas_name = CORRECT_ATLAS_NAME(atlas_name)
        if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
            data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        else:
            data_dir = f"{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}"
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        if not os.path.exists(f'data/{data_dir}/raw.pt'):
            fc_subs = [fn.split('_')[subn_p] for fn in os.listdir(fc_root)]
            fc_subs = np.unique(fc_subs)
            sc_subs = [fn.split('_')[subn_p] for fn in os.listdir(sc_root)]
            subs = np.intersect1d(fc_subs, sc_subs)
            self.all_sc = {}
            self.all_fc = {}
            self.label_name = []
            self.sc_common_rname = None
            for fn in tqdm(os.listdir(sc_root), desc='Load SC'):
                subn = fn.split('_')[subn_p]
                if subn in subs:
                    sc, rnames, _ = load_sc(f"{sc_root}/{fn}", atlas_name)
                    if self.sc_common_rname is None: self.sc_common_rname = rnames
                    if self.sc_common_rname is not None: 
                        _, rid, _ = np.intersect1d(rnames, self.sc_common_rname, return_indices=True)
                        self.all_sc[subn] = sc[rid, :][:, rid]
                    else:
                        self.all_sc[subn] = sc
            self.fc_common_rname = None
            # compute FC in getitem
            self.data = {'bold': [], 'subject': [], 'label': [], 'winid': []}
            for fn in tqdm(os.listdir(bold_root), desc='Load BOLD'):
                if fn.split('_')[subn_p] in subs:
                    bolds, rnames, fn = bold2fc(f"{bold_root}/{fn}", self.fc_winsize, fc_winoverlap, onlybold=True)
                    subn = fn.split('_')[subn_p]
                    if self.fc_common_rname is None: self.fc_common_rname = rnames
                    if self.fc_common_rname is not None: 
                        _, rid, _ = np.intersect1d(rnames, self.fc_common_rname, return_indices=True)
                        bolds = [b[rid] for b in bolds]
                
                    label = Path(fn).stem.split('_')[subtask_p]
                    if dname in ['adni', 'oasis']:
                        if label not in LABEL_REMAP[dname]: continue
                        label = LABEL_REMAP[dname][label]
                    if label not in self.label_name: self.label_name.append(label)
                    self.data['bold'].extend(bolds) # N x T
                    self.data['subject'].extend([subn for _ in bolds])
                    self.data['label'].extend([self.label_name.index(label) for _ in bolds])
                    self.data['winid'].extend([i for i in range(len(bolds))])

            if self.sc_common_rname is not None and self.fc_common_rname is not None:
                self.sc_common_rname = [rn.strip() for rn in self.sc_common_rname]
                self.fc_common_rname = [rn.strip() for rn in self.fc_common_rname]
                common_rname, sc_rid, fc_rid = np.intersect1d(self.sc_common_rname, self.fc_common_rname, return_indices=True)
                for sub in self.all_sc:
                    self.all_sc[sub] = self.all_sc[sub][:, sc_rid][sc_rid, :]
                for i in range(len(self.data['subject'])):
                    self.data['bold'][i] = self.data['bold'][i][fc_rid]
                self.sc_common_rname = common_rname
                self.fc_common_rname = common_rname
            self.data['all_sc'] = self.all_sc
            self.data['label_name'] = self.label_name
            torch.save(self.data, f'data/{data_dir}/raw.pt')
        
        self.data = torch.load(f'data/{data_dir}/raw.pt')
        self.all_sc = self.data['all_sc']
        self.adj_type = adj_type
        self.node_attr = node_attr
        self.atlas_name = atlas_name
        self.subject = np.array(self.data['subject'])
        # self.data['label'] = np.array(self.data['label'])
        self.label_names = list(self.data['label_name'])
        self.data_subj = np.unique(self.subject)
        self.node_num = len(self.data['bold'][0])
        self.cached_data = [None for _ in range(len(self.subject))]
        self.label_remap = None
        if 'task-rest' in self.data['label_name'] or 'task-REST' in self.data['label_name']:
            restli = [i for i, l in enumerate(self.data['label_name']) if 'rest' in l.lower()]
            assert len(restli) == 1, self.data['label_name']
            restli = restli[0]
            nln = list(self.data['label_name'])
            nln[0] = self.data['label_name'][restli]
            nln[restli] = self.data['label_name'][0]
            self.data['label_name'] = nln
            self.label_remap = {restli: 0, 0: restli}

        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==subj]['Sex']).item()
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            self.sex_label = {'M': 0, 'F': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: str(np.unique(meta_data[meta_data['Subject']==subj]['Age']).item())
            for subj in self.data_subj if subj in list(meta_data['Subject'])}
            self.subj2age = {
                k: float(v) if '-' not in v else np.array(v.split('-')).astype(np.int32).mean()
            for k, v in self.subj2age.items()}
            self.subj2age = {
                k: torch.tensor(v).float().reshape(1, 1)
            for k, v in self.subj2age.items()}
            if len(self.subj2age) > 0:
                # self.age_max = np.max(list(self.subj2age.values()))
                # self.age_min = np.min(list(self.subj2age.values()))
                # self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
                self.subj2age = {k: v if v <= 150 else v/12 for k, v in self.subj2age.items()}
        else:
            self.subj2sex = {}
            self.subj2age = {}
        
        print("Data num", len(self), "BOLD shape (N x T)", self.data['bold'][0].shape, "Label name", self.data['label_name'])
        if self.transform is not None:
            processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
            if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
                for _ in tqdm(self, desc='Processing'):
                    pass
                
                torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
            self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
        for _ in tqdm(self, desc='Preloading'):
            pass
        
        self.dname_list = None
        self.nclass_list = [len(self.label_names)]
        
    def __getitem__(self, index):
        if self.cached_data[index] is None:
            subjn = self.subject[index]
            fc = torch.corrcoef(self.data['bold'][index])
            sc = self.all_sc[subjn]
            edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            edge_index_sc = torch.stack(torch.where(sc > self.sc_th))
            if self.adj_type == 'FC':
                edge_index = edge_index_fc
                # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            else:
                edge_index = edge_index_sc
                # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            if self.node_attr=='FC':
                x = fc
            elif self.node_attr=='BOLD':
                x = self.data['bold'][index]
            elif self.node_attr=='SC':
                x = sc
            elif self.node_attr=='ID':
                x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                'edge_index': edge_index,
                'x': x,
                'y': self.data['label'][index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([[-1]]).float(),
                'edge_index_fc': edge_index_fc,
                'edge_index_sc': edge_index_sc
            }
            if self.transform is not None:
                new_data = self.transform(Data.from_dict(data))
                # self.cached_data[index] = new_data
                for key in new_data:
                    data[key] = new_data[key]
                    
            adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            data['adj_fc'] = adj_fc[None]
            data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = Data.from_dict(data)
        data = self.cached_data[index]
        if self.label_remap is not None:
            if data.y in self.label_remap:
                data.y = self.label_remap[data.y]
        return data

    def __len__(self):
        return len(self.cached_data)


def load_fc(fpath):
    mat = pd.read_csv(fpath)
    mat = torch.from_numpy(mat[:, 1:].astype(np.float32))
    rnames = mat[:, 0]
    return mat, rnames, fpath.split('/')[-1]

def load_sc(path, atlas_name):
    if not path.endswith('.mat') and not path.endswith('.txt'):
        matfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]
        txtfns = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
        return load_sc(matfns[0] if len(matfns) > 0 else txtfns[0], atlas_name)
    if path.endswith('.mat'):
        fpath = f"{path}"
        sc_mat = loadmat(fpath)
        mat = sc_mat[f"{atlas_name.lower().replace('_','')}_sift_radius2_count_connectivity"]
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = sc_mat[f"{atlas_name.lower().replace('_','')}_region_labels"]
    elif path.endswith('.txt'):
        fpath = f"{path}"
        mat = np.loadtxt(fpath)
        mat = torch.from_numpy(mat.astype(np.float32))
        mat = (mat + mat.T) / 2
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        rnames = None
    return mat, rnames, path.split('/')[-1]

def bold2fc(path, winsize, overlap, onlybold=False):
    if not path.endswith('.txt'):
        bold_pd = pd.read_csv(path) if not path.endswith('.tsv') else pd.read_csv(path, sep='\t')
        if isinstance(np.array(bold_pd)[0, 0], str):
            rnames = list(bold_pd.columns[1:])
            bold = torch.from_numpy(np.array(bold_pd)[:, 1:]).float().T
        else:
            rnames = list(bold_pd.columns)
            bold = torch.from_numpy(np.array(bold_pd)).float().T
    else:
        rnames = None
        bold = torch.from_numpy(np.loadtxt(path)).float().T
    # bold = bold[torch.logical_not(bold.isnan().any(dim=1))]
    # rnames = [rnames[i] for i in torch.where(torch.logical_not(bold.isnan().any(dim=1)))[0]]
    # bold = (bold - bold.min()) / (bold.max() - bold.min())
    timelen = bold.shape[1]
    steplen = int(winsize*(1-overlap))
    fc = []
    if onlybold:
        bolds = []
    for tstart in range(0, timelen, steplen):
        b = bold[:, tstart:tstart+winsize]
        if b.shape[1] < winsize: 
            # b = bold[:, -winsize:]
            b = torch.cat([b, torch.zeros([b.shape[0], winsize-b.shape[1]], dtype=b.dtype)], dim=1)
        if onlybold: 
            bolds.append(b)
            continue
        fc.append(torch.corrcoef(b))#.cpu()
    if onlybold:
        return bolds, rnames, path.split('/')[-1]
    fc = torch.stack(fc)
    return fc, rnames, path.split('/')[-1]

def CORRECT_ATLAS_NAME(n):
    if n == 'Brainnetome_264': return 'Brainnetome_246'
    if 'Shaefer_' in n: return n.replace('Shaefer', 'Schaefer')
    return n

def Schaefer_SCname_match_FCname(scns, fcns):
    '''
    TODO: Align Schaefer atlas region name of SC and FC
    '''
    match = []
    def get_overlap(s1, s2):
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
        return s1[pos_a:pos_a+size]
    
    for fcn in fcns:
        fcn = fcn.replace('17Networks_', '')
        fcn_split = fcn.split('_')
        sc_overlap_len = []
        for scn in scns:
            scn_split = scn.split('_')
            if scn_split[0] != fcn_split[0] or scn_split[-1] != fcn_split[-1]:
                continue
            sc_overlap_len.append(sum([len(get_overlap(scn_split[i], fcn_split[i])) for i in range(1, len(scn_split)-1)]))
        match.append()

    return match

def tsne_spdmat(mats):
    tril_ind = torch.tril_indices(mats.shape[1], mats.shape[2])
    X = mats[:, tril_ind[0], tril_ind[1]]
    X_embedded = TSNE(n_components=2, random_state=142857).fit_transform(X.numpy())
    return X_embedded

def ttest_fc(fcs1, fcs2, thr=0.05):
    from scipy import stats
    print(fcs1.shape, fcs2.shape)
    significant_fc = []
    ps = []
    for i in trange(fcs1.shape[1]):
        for j in range(i+1, fcs1.shape[2]):
            a = fcs1[:, i, j].numpy()
            b = fcs2[:, i, j].numpy()
            p = stats.ttest_ind(a, b).pvalue
            if p < thr: 
                significant_fc.append([i, j])
                ps.append(p)
    significant_fc = torch.LongTensor(significant_fc)
    ps = torch.FloatTensor(ps)
    print(significant_fc.shape)
    return significant_fc, ps

import re
class Dataset_PPMI_ABIDE(Dataset):
    def __init__(self, atlas_name='AAL_116', # multi-atlas not available 
                 dname='ppmi',
                node_attr = 'FC', adj_type = 'FC',
                transform = None,
                fc_winsize = 137, # not implement
                fc_winoverlap = 0, # not implement
                fc_th = 0.5,
                sc_th = 0.1, **kargs):
        super(Dataset_PPMI_ABIDE, self).__init__()
        self.adj_type = adj_type
        self.transform = transform
        self.node_attr = node_attr
        self.atlas_name = atlas_name
        self.fc_th = fc_th
        self.sc_th = sc_th
        self.fc_winsize = fc_winsize
        self.node_num = 116
        self.label_remap = None
        self.root_dir = DATAROOT[dname]
        self.dname = dname
        self.data = []
        self.labels = []
        self.data_path = []
        self.subject = []
        self.label_names = [None for _ in range(4)]
        
        default_fc_th = 0.5
        default_sc_th = 0.1
        if self.fc_th == default_fc_th and self.sc_th == default_sc_th:
            data_dir = f'{dname}-{atlas_name}-BOLDwin{fc_winsize}'
        else:
            data_dir = f"{dname}-{atlas_name}-BOLDwin{fc_winsize}-FCth{str(self.fc_th).replace('.', '')}SCth{str(self.sc_th).replace('.', '')}"
        os.makedirs(f'data/{data_dir}', exist_ok=True)
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    self.data_path.append(file_path)
                    label, label_name = self.get_label(subdir)
                    self.labels.append(label)
                    self.label_names[label] = label_name
                    self.subject.append(subdir.split('/')[-1])
        self.label_names = [l for l in self.label_names if l is not None]
        self.cached_data = [None for _ in range(len(self.data_path))]
        self.data_subj = np.unique(self.subject)
        if os.path.exists(f'../data/meta_data/{dname.upper()}_metadata.csv'):
            meta_data = pd.read_csv(f'../data/meta_data/{dname.upper()}_metadata.csv')
            self.subj2sex = {
                subj: np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Sex']).item()
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            self.sex_label = {'M': 0, 'F': 1}
            self.subj2sex = {k: self.sex_label[v] for k, v in self.subj2sex.items()}
            self.subj2age = {
                subj: torch.tensor(np.unique(meta_data[meta_data['Subject']==int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0])]['Age']).item()).float().reshape(1, 1)
            for subj in self.data_subj if int(re.findall(r"[-+]?\d*\.\d+|\d+", subj)[0]) in list(meta_data['Subject'])}
            # if len(self.subj2age) > 0:
            #     self.age_max = np.max(list(self.subj2age.values()))
            #     self.age_min = np.min(list(self.subj2age.values()))
            #     self.subj2age = {k: (v-self.age_min)/(self.age_max-self.age_min) for k, v in self.subj2age.items()}
        else:
            self.subj2sex = {}
            self.subj2age = {}
        if self.transform is not None:
            processed_fn = f'processed_adj{self.adj_type}x{self.node_attr}_FCth{self.fc_th}SCth{self.sc_th}_{type(self.transform).__name__}{self.transform.k}'.replace('.', '')
            if not os.path.exists(f'data/{data_dir}/{processed_fn}.pt'):
                for _ in tqdm(self, desc='Processing'):
                    pass
                
                torch.save(self.cached_data, f'data/{data_dir}/{processed_fn}.pt')
            self.cached_data = torch.load(f'data/{data_dir}/{processed_fn}.pt')
        
        for _ in tqdm(self, desc='Preload data'):
            pass
        self.nclass_list = [len(self.label_names)]
        print("Data num", len(self), "Label name", self.label_names)
            
    def get_label(self, subdir):
        if 'control' in subdir:
            return 0, 'CN'
        elif 'patient' in subdir:
            return 1, 'Autism' if 'ABIDE' in subdir else 'PD'
        elif 'prodromal' in subdir:
            return 2, 'ppmi-2'
        elif 'swedd' in subdir:
            return 3, 'ppmi-3'
        else:
            assert False, subdir
        
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, index):
        # label = self.labels[index]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        if self.cached_data[index] is None:
            features = loadmat(self.data_path[index])['data'].T
            features = torch.from_numpy(features).float()
            x = torch.nan_to_num(features)
            fc = torch.corrcoef(x)
            subjn = self.subject[index]
            edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            # edge_index_sc = None
            edge_index_sc = edge_index_fc
            if self.adj_type == 'FC':
                edge_index = edge_index_fc
                # adj = torch.sparse_coo_tensor(indices=edge_index_fc, values=fc[edge_index_fc[0], edge_index_fc[1]], size=(self.node_num, self.node_num))
            else:
                assert False, "Not implement"
                # adj = torch.sparse_coo_tensor(indices=edge_index_sc, values=sc[edge_index_sc[0], edge_index_sc[1]], size=(self.node_num, self.node_num))
            if self.node_attr=='FC':
                x = fc
            elif self.node_attr=='BOLD':
                x = x[:, :self.fc_winsize]
                if x.shape[1] < self.fc_winsize: 
                    x = torch.cat([x, torch.zeros(x.shape[0], self.fc_winsize-x.shape[1])], 1)
                    
            elif self.node_attr=='SC':
                assert False, "Not implement"
            elif self.node_attr=='ID':
                x = torch.arange(self.node_num).float()[:, None]
        
            x[x.isnan()] = 0
            x[x.isinf()] = 0
            data = {
                'edge_index': edge_index,
                'x': x,
                'y': self.labels[index],
                'sex': self.subj2sex[subjn] if subjn in self.subj2sex else -1,
                'age': self.subj2age[subjn] if subjn in self.subj2age else torch.tensor([[-1]]).float(),
                'edge_index_fc': edge_index_fc,
                'edge_index_sc': edge_index_sc
            }
            if self.transform is not None:
                new_data = self.transform(Data.from_dict(data))
                # self.cached_data[index] = new_data
                for key in new_data:
                    data[key] = new_data[key]
                    
            adj_fc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
            adj_sc = torch.zeros(x.shape[0], x.shape[0]).bool()
            adj_sc[edge_index_sc[0], edge_index_sc[1]] = True
            adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            adj_sc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
            data['adj_fc'] = adj_fc[None]
            data['adj_sc'] = adj_sc[None]
            
            self.cached_data[index] = Data.from_dict(data)
        data = self.cached_data[index]
        if self.label_remap is not None:
            if data.y in self.label_remap:
                data.y = self.label_remap[data.y]
        return data

DATASET_CLASS = {
    'adni': NeuroNetworkDataset,
    'oasis': NeuroNetworkDataset,
    'hcpa': NeuroNetworkDataset,
    'ukb': NeuroNetworkDataset,
    'hcpya': NeuroNetworkDataset,
    'ppmi': Dataset_PPMI_ABIDE,
    'abide': Dataset_PPMI_ABIDE,
    'neurocon': Dataset_PPMI_ABIDE,
    'taowu': Dataset_PPMI_ABIDE
}

def dataloader_generator(batch_size=4, num_workers=8, nfold=0, total_fold=5, dataset=None, testset='None', **kargs):
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=142857)
    if dataset is None:
        dataset = DATASET_CLASS[kargs['dname']](**kargs)
    if isinstance(testset, str):
        if testset != 'None':
            # if testset != 'oasis' and testset != 'adni' and kargs['dname'] != 'adni' and kargs['dname'] != 'oasis':
            #     del kargs['dname']
            #     testset = DATASET_CLASS[kargs['dname']](dname=testset, **kargs)
            # else:
            del kargs['atlas_name'], kargs['dname']
            atlas_name = {'adni': 'AAL_116', 'oasis': 'D_160', 'ukb': 'Gordon_333', 'hcpa': 'Gordon_333'}
            testset = DATASET_CLASS[kargs['dname']](dname=testset, atlas_name=atlas_name[testset], **kargs)
    all_subjects = dataset.data_subj
    train_index, index = list(kf.split(all_subjects))[nfold]
    train_subjects = [all_subjects[i] for i in train_index]
    subjects = [all_subjects[i] for i in index]
    # Filter dataset based on training and validation subjects
    train_data = [di for di, subj in enumerate(dataset.subject) if subj in train_subjects]
    data = [di for di, subj in enumerate(dataset.subject) if subj in subjects]
    print(f'Fold {nfold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    valid_dataset = torch.utils.data.Subset(dataset, data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if testset == 'None':
        return train_loader, loader, dataset
    else:
        _, index = list(kf.split(testset.data_subj))[nfold]
        subjects = [testset.data_subj[i] for i in index]
        data = [di for di, subj in enumerate(testset.subject) if subj in subjects]
        print(f'Fold {nfold + 1}, Test {len(subjects)} subjects, len(test_data)={len(data)}')
        test_dataset = torch.utils.data.Subset(testset, data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, loader, dataset, test_loader, testset

import copy

def merge_datasets(datasets, dnames):
    default_type = 'task-REST'
    all_label_names = [datasets[dname].label_names for dname in dnames]

    label_is_subset = np.array([[set(all_label_names[i]).issubset(set(all_label_names[j])) if i != j else False for i in range(len(all_label_names))] for j in range(len(all_label_names))])
    # print(label_is_subset)
    label_top_node = list(np.where(np.logical_not(label_is_subset).all(0))[0])
    # print(label_top_node)
    label_is_subset = [np.where(label_is_subset[i])[0] for i in range(len(label_is_subset))]
    # print(label_is_subset)
    merged = []
    def merge_label(i):
        out = [i]
        merged.append(i)
        for l in label_is_subset[i]:
            if l in merged: continue
            m = merge_label(l)
            merged.extend(m)
            out.extend(m)
        return out
    merge_label_id = []
    for i in label_top_node:
        merge_label_id.append(merge_label(i))
    
    dataset2merged = {}
    for ki, k in enumerate(dnames):
        for i in range(len(merge_label_id)):
            if ki in merge_label_id[i]: break
        dataset2merged[k] = i
    print(all_label_names)
    print(merge_label_id)
    # print(dataset2merged)
    dataset_is_tfmri = np.array([np.array(['task' in n for n in names]).all() for names in all_label_names])
    if dataset_is_tfmri.any() and not dataset_is_tfmri.all():
        id_put_default_label = np.where([np.array([default_type in n for n in names]).any() for names in all_label_names])[0][0]
        id_put_default_label = dataset2merged[dnames[id_put_default_label]]
        use_default = True
    else:
        use_default = False
    
    merged_dataset = copy.deepcopy(datasets[dnames[0]])
    new_data_list = []
    new_nclass = [len(datasets[dnames[di[0]]].label_names) + 1 for di in merge_label_id]
    new_label_names = [['N/A'] + datasets[dnames[di[0]]].label_names for di in merge_label_id]
    merged_dataset.nclass_list = new_nclass
    merged_dataset.label_names = new_label_names
    # print(merged_dataset.nclass_list)
    # print(merged_dataset.label_names)
    for di, dname in enumerate(dnames):
        dset = datasets[dname]
        # new_data = []
        for i, d in enumerate(dset.cached_data):   
            d = d.clone()
            new_y = torch.zeros(1, len(merge_label_id)).long()
            new_y[0, dataset2merged[dname]] = d.y + 1
            if not dataset_is_tfmri[di] and use_default: new_y[0, id_put_default_label] = 1
            d.y = new_y
            new_data_list.append(d) 

        # new_data_list.extend(new_data)
    merged_dataset.cached_data = new_data_list
    merged_dataset.dnames = dnames
    # print(len(merged_dataset), merged_dataset[0])
    return merged_dataset, dataset2merged
    
def multidataloader_generator(
    # val_dname='adni',
    dname_list=['adni','hcpa','hcpya','abide','ppmi','taowu','neurocon'], 
    datasets={'adni': None,'hcpa': None,'hcpya': None,'abide': None,'ppmi': None,'taowu': None,'neurocon': None}, 
    total_fold={'adni': 5,'hcpa': 5,'hcpya': 5,'abide': 10,'ppmi': 10,'taowu': 10,'neurocon': 10},
    batch_size=4, num_workers=8, nfold=0, **kargs
):
    # assert val_dname in dname_list 
    train_index_list = []
    valid_index_list = []
    loaders = {}
    pre_data_n = 0
    for dname in dname_list:
        kargs['dname'] = dname
        kf = KFold(n_splits=total_fold[dname], shuffle=True, random_state=142857)
        dataset = datasets[dname]
        if dataset is None:
            dataset = DATASET_CLASS[dname](**kargs)
        datasets[dname] = dataset
        all_subjects = dataset.data_subj
        train_index, index = list(kf.split(all_subjects))[min(nfold, total_fold[dname]-1)]
        train_subjects = [all_subjects[i] for i in train_index]
        subjects = [all_subjects[i] for i in index]
        train_data = [di + pre_data_n for di, subj in enumerate(dataset.subject) if subj in train_subjects]
        data = [di + pre_data_n for di, subj in enumerate(dataset.subject) if subj in subjects]
        print(f'Fold {nfold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
        train_index_list.extend(train_data)
        valid_index_list.append(data)
        # if val_dname == dname: valid_index_list = data
        pre_data_n += len(dataset)
        
    merged_dataset, dname2tokenid = merge_datasets(datasets, dname_list)
    merged_dataset.dname2tokenid = dname2tokenid
    # merged_dataset.val_dname = val_dname
    train_dataset = torch.utils.data.Subset(merged_dataset, train_index_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    for di, dname in enumerate(dname_list):
        valid_dataset = torch.utils.data.Subset(merged_dataset, valid_index_list[di])
        loaders[dname] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, loaders, merged_dataset, datasets


if __name__ == '__main__':
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    # dname = 'hcpya'
    # dname_list = ['adni', 'hcpa', 'hcpya']
    dname = 'taowu'
    tl, vl, ds = dataloader_generator(dname=dname, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)
    print(len(ds.subj2sex))
    dname = 'neurocon'
    tl, vl, ds = dataloader_generator(dname=dname, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)
    print(len(ds.subj2sex))
    dname = 'ppmi'
    tl, vl, ds = dataloader_generator(dname=dname, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)
    print(len(ds.subj2sex))
    dname = 'abide'
    tl, vl, ds = dataloader_generator(dname=dname, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)
    print(len(ds.subj2sex))
    exit()
    dname_list = ['abide','ppmi','taowu','neurocon']
    org_ds = {'adni': None,'hcpa': None,'hcpya': None,'abide': None,'ppmi': None,'taowu': None,'neurocon': None}
    for i in range(5):
        tl, vl, ds, org_ds = multidataloader_generator(dname_list=dname_list, nfold=i, datasets=org_ds, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)
        # print('fold', i, len(tl), len(vl), len(ds))
        print(ds.dname2tokenid)
        exit()
        for d in tqdm(tl, desc=f'fold {i} train, {len(ds)}'):
            pass
            # print(d)
            # break
        for d in tqdm(vl, desc=f'fold {i} val, {len(ds)}'):
            pass
            # print(d)
            # break
    # from models.graphormer import ShortestDistance
    # from models.nagphormer import NAGdataTransform
    # tl, vl, ds = dataloader_generator(dname=dname, atlas_name='AAL_116', node_attr='FC', fc_winsize=500)#, transform=NAGdataTransform(), transform=NeuroDetourNode(k=5, node_num=333)
    # for d in tl:
    #     print(d.age, d.sex)
    #     break
    # for d in vl:
    #     print(d.age, d.sex)
    #     break
    