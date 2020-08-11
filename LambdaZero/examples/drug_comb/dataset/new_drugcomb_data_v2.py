import time
from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Data, InMemoryDataset
from pubchempy import Compound
import urllib.request
import ssl
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm



def _cid_to_smiles(_drugcomb_data, raw_dir):
    unique_cid = set(_drugcomb_data['drug_row_cid'].unique()).union(_drugcomb_data['drug_col_cid'].unique())
    unique_cid.remove(0)
    unique_cid = list(unique_cid)

    dict_path = os.path.join(raw_dir, "cid_to_smiles_dict.npy")
    if os.path.exists(dict_path):
        cid_to_smiles_dict = np.load(dict_path, allow_pickle=True).item()
    else:
        print('recovering smiles, only happens the first time...')
        cid_to_smiles_dict = {}
        for cid in tqdm(unique_cid):
            try:
                cid_to_smiles_dict[int(cid)] = Compound.from_cid(int(cid)).isomeric_smiles
            except:
                print("error with cid", cid)
                pass
        np.save(dict_path, cid_to_smiles_dict)
    return cid_to_smiles_dict

def _cell_line_to_idx(_drugcomb_data, raw_dir):
    cell_lines_names = _drugcomb_data[["cell_line_name"]].to_numpy()[:, 0]
    cell_line_to_idx = {na:i for i,na in enumerate(cell_lines_names)}
    return cell_line_to_idx

def _get_fingerprint(smiles, radius, n_bits):
    if smiles == 'none':
        return np.array([-1] * n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, n_bits))
    except Exception as ex:
        return np.array([-1] * n_bits)


def _get_nodes(_drugcomb_data, cid_to_smiles_dict, fp_radius, fp_bits):
    _drugcomb_data['drug_row_smiles'] = _drugcomb_data['drug_row_cid'].apply(
        lambda cid: cid_to_smiles_dict[cid] if cid in cid_to_smiles_dict.keys() else -1)

    _drugcomb_data['drug_col_smiles'] = _drugcomb_data['drug_col_cid'].apply(
        lambda cid: cid_to_smiles_dict[cid] if cid in cid_to_smiles_dict.keys() else -1)

    # Computing fingerprints
    cid_to_fp_dict = {cid: _get_fingerprint(cid_to_smiles_dict[cid], fp_radius, fp_bits)
                      for cid in cid_to_smiles_dict.keys()}
    nodes = pd.DataFrame.from_dict(cid_to_fp_dict, orient='index')
    cid_to_idx_dict = {nodes.index[i]: i for i in range(len(nodes))}
    nodes.reset_index(drop=True, inplace=True)
    return nodes, cid_to_idx_dict


def _append_cid(_drugcomb_data, _summary_data):
    first = _drugcomb_data[['drug_row', 'drug_row_cid']].rename(columns={'drug_row': 'name', 'drug_row_cid': 'cid'})
    scnd = _drugcomb_data[['drug_col', 'drug_col_cid']].rename(columns={'drug_col': 'name', 'drug_col_cid': 'cid'})
    uniques = first.append(scnd).dropna().drop_duplicates().values.tolist()
    name_to_cid = {tpl[0]: tpl[1] for tpl in uniques}

    _summary_data['drug_row_cid'] = _summary_data['drug_row'].apply(
        lambda drug: name_to_cid[drug] if drug in name_to_cid else -1)
    _summary_data['drug_col_cid'] = _summary_data['drug_col']. \
        apply(lambda drug: name_to_cid[drug]
    if drug in name_to_cid else -1)

    _summary_data = _summary_data[_summary_data['drug_row_cid'] != -1]
    _summary_data = _summary_data[_summary_data['drug_col_cid'] != -1]
    return _summary_data

def _get_ddi_edges(_drugcomb_data, cid_to_idx, cell_line_to_idx,properties):
    print('Processing drug drug interaction edges..')
    # Drop rows that do not have second drug
    _drugcomb_data = _drugcomb_data.dropna(axis=0, subset=['drug_col'])

    _drugcomb_data['drug_row_idx'] = _drugcomb_data['drug_row_cid'].apply(lambda cid: cid_to_idx[cid])
    _drugcomb_data['drug_col_idx'] = _drugcomb_data['drug_col_cid'].apply(lambda cid: cid_to_idx[cid])
    is_pairs_of_diff_drugs = _drugcomb_data['drug_row_idx'] != _drugcomb_data['drug_col_idx']
    _drugcomb_data = _drugcomb_data[is_pairs_of_diff_drugs]

    ddi_edge_idx = _drugcomb_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
    ddi_edge_attr = _drugcomb_data[properties].to_numpy()

    cell_line_names = _drugcomb_data[["cell_line_name"]].to_numpy()[:,0]
    ddi_edge_classes = np.asarray([cell_line_to_idx[na] for na in cell_line_names],dtype=np.int)
    return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes

class NewDrugComb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=3, n_laplace_feat=256, **kwargs):

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.n_laplace_feat = n_laplace_feat

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/NewDrugComb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[1])


    @property
    def raw_file_names(self):
        return [
            'drugcomb_data_v1.4.csv',
            'summary_table_v1.4.csv'
        ]

    @property
    def processed_file_names(self):
        return ['drugcomb_data.pt', "summary_table.pt"]

    def download(self):
        urls = [
            'https://drugcomb.fimm.fi/download/drugcomb_data_v1.4.csv',
            'https://drugcomb.fimm.fi/download/summary_table_v1.4.csv'
        ]

        chunk_size = 16 * 1024
        for url in urls:
            file_name = url.split('/')[-1]
            gcontext = ssl.SSLContext()
            with urllib.request.urlopen(url, context=gcontext) as f:
                with open(os.path.join(self.raw_dir, file_name), "w") as file:
                    while True:
                        file_chunk = f.read(chunk_size).decode('utf-8')
                        if not file_chunk:
                            break
                        file.write(file_chunk)

    def process(self):
        # Load dataframes
        _drugcomb_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), low_memory=False)
        self.cid_to_smiles = _cid_to_smiles(_drugcomb_data, self.raw_dir)
        self.cell_line_to_idx = _cell_line_to_idx(_drugcomb_data, self.raw_dir)

        nodes, cid_to_idx = _get_nodes(_drugcomb_data, self.cid_to_smiles, self.fp_radius, self.fp_bits)

        # todo: save raw_data_table
        # ddi_edge_idx, ddi_edge_attr, ddi_edge_classes = _get_ddi_edges(self._drugcomb_data, cid_to_idx,
        #                                                                ['conc_r', 'conc_c', 'inhibition'])
        torch.save(torch.tensor([0]), self.processed_paths[0])

        _summary_table = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]), low_memory=False)
        _summary_table = _append_cid(_drugcomb_data, _summary_table)
        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes = _get_ddi_edges(_summary_table, cid_to_idx,
                                                                self.cell_line_to_idx, ["css_row","css_col", "css"])

        data = Data(x_drugs=torch.tensor(nodes.to_numpy(), dtype=torch.float))

        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.float)

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[1])




if __name__ == '__main__':

    dataset = NewDrugComb()
    print(dataset[0])

# todo: log transform concentrations

# ['block_id' 'conc_r' 'conc_c' 'inhibition' 'drug_row' 'drug_col'
# 'conc_r_unit' 'conc_c_unit' 'cell_line_name' 'drug_row_cid'
# 'drug_col_cid' 'cellosaurus_accession' 'study_name']

# ['block_id' 'drug_row' 'drug_col' 'cell_line_name' 'conc_r_unit'
# 'conc_c_unit' 'css' 'synergy_zip' 'synergy_bliss' 'synergy_loewe'
# 'synergy_hsa' 'ic50_row' 'ic50_col' 'ri_row' 'ri_col' 'css_row' 'css_col'
# 'S']
