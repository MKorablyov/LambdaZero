from torch_geometric.data import Data, InMemoryDataset, download_url
from recover.utils import get_project_root
import pubchempy as pcp
from cmapPy.pandasGEXpress.parse import parse
from rdkit.Chem import AllChem
from pubchempy import Compound
import pickle
import pandas as pd
import numpy as np
import torch
import os
import gzip
import shutil
from tqdm import tqdm

def _get_fingerprint(smile, radius, n_bits):
    if smile == 'none':
        return np.array([-1] * n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
    except Exception as ex:
        return np.array([-1] * n_bits)

def get_fingerprints(cids, cid_to_smiles, radius, n_bits):
    unique_cids = np.unique(cids)
    cid_to_fp_dict = {}

    for cid in unique_cids:
        if cid not in cid_to_smiles:
            cid_to_fp_dict[cid] = np.array([-1] * n_bits)
        else:
            cid_to_fp_dict[cid] = _get_fingerprint(cid_to_smiles[cid], radius, n_bits)

    return np.ndarray([cid_to_fp_dict[cid] for cid in cids])

class L1000(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, name='LINCS/'):
        """
        Dataset object for the LINCS L1000 dataset.
        """

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        super().__init__(os.path.join(get_project_root(), name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        print("L1000 dataset loaded.")

    @property
    def raw_file_names(self):
        """ The first 5 files correspond to phase 2, the last 3 to phase 1"""
        return [
            'Level5_COMPZ_n118050x12328_2017-03-06.gctx',
            'sig_info_2017-03-06.txt',
            'gene_info_2017-03-06.txt',
            'pert_info_2017-03-06.txt',
            'cell_info_2017-04-28.txt',
            'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx',
            'GSE92742_Broad_LINCS_sig_info.txt',
            'GSE92742_Broad_LINCS_pert_info.txt'
        ]

    @property
    def processed_file_names(self):
        return ['l1000_data.pt']

    def download(self):
        urls = {
            0: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5FLevel5'
            '%5FCOMPZ%5Fn118050x12328%5F2017%2D03%2D06%2Egctx%2Egz',
            1: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fsig%5F'
            'info%5F2017%2D03%2D06%2Etxt%2Egz',
            2: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fgene%5'
            'Finfo%5F2017%2D03%2D06%2Etxt%2Egz',
            3: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fpert%5'
            'Finfo%5F2017%2D03%2D06%2Etxt%2Egz',
            4: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fcell%5'
            'Finfo%5F2017%2D04%2D28%2Etxt%2Egz',
            5: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5FLevel5'
            '%5FCOMPZ%2EMODZ%5Fn473647x12328%2Egctx%2Egz',
            6: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fsig%5F'
            'info%2Etxt%2Egz',
            7: 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fpert%5'
            'Finfo%2Etxt%2Egz'
        }

        for idx, url in urls.items():
            download_url(url, self.raw_dir)
            with gzip.open(os.path.join(self.raw_dir, url.split('/')[-1]), 'rb') as f_in:
                with open(os.path.join(self.raw_dir, self.raw_file_names[idx]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def process(self):
        # Load metadata
        self.sig_info, self.landmark_gene_list = self.load_metadata()
        self.expr_data = pd.concat([self.load_expr_data("phase1"), self.load_expr_data("phase2")], sort=False)

        data_df = pd.concat((self.expr_data, self.sig_info), axis=1)  # Concat with metadata

        data = self.build_data(self.filter_df(data_df))

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def filter_df(self, data_df):
        '''
        Method allows classes inheriting from L1000 to easily select which
        values to include in the data object
        '''
        return data_df

    def build_data(self, data_df):
        # Extract numpy arrays
        gene_expr = data_df[data_df.columns[:len(self.landmark_gene_list)]].to_numpy()
        cid = data_df['cid'].to_numpy().astype(np.int32)
        #fingerprints = get_fingerprints(cid, self.get_smiles(cid), self.fp_radius, self.fp_bits)
        cell_id = data_df['cell_id'].to_numpy()
        cell_id_int = pd.factorize(data_df['cell_id'])[0]
        dose = data_df['pert_idose_value'].to_numpy()
        incub_time = data_df['pert_itime_value'].to_numpy()

        # Create data object
        return Data(gene_expr=torch.tensor(gene_expr, dtype=torch.float),
                    cid=torch.tensor(cid, dtype=torch.long),
                    cell_id=cell_id,
                    cell_id_int=torch.tensor(cell_id_int, dtype=torch.long),
                    dose=torch.tensor(dose, dtype=torch.float),
                    incub_time=torch.tensor(incub_time, dtype=torch.float))

    def load_metadata(self):
        # Get list of landmark genes. Gene info and cell info are the same for both phases
        gene_info = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[2]), sep="\t")
        landmark_gene_list = gene_info[gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        # Load pert_info
        pert_info_with_cid_path = os.path.join(self.raw_dir, "pert_info_with_cid.txt")
        if os.path.isfile(pert_info_with_cid_path):
            pert_info = pd.read_csv(pert_info_with_cid_path, index_col="pert_id")
        else:
            print("Retrieving cids from PubChem, only happens the first time...")
            # Load both phases
            pert_info_1 = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[3]), sep="\t",
                                    index_col="pert_id", usecols=["pert_id", "canonical_smiles"])
            pert_info_2 = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[7]), sep="\t",
                                      index_col="pert_id", usecols=["pert_id", "canonical_smiles"])
            pert_info = pd.concat([pert_info_1, pert_info_2])
            # Remove duplicate indices
            pert_info = pert_info.loc[~pert_info.index.duplicated(keep='first')]

            pert_info = pert_info[pert_info['canonical_smiles'] != '-666']
            pert_info['cid'] = - 1
            for i in tqdm(pert_info.index):
                try:
                    pert_info.at[i, 'cid'] = pcp.get_compounds(pert_info.at[i, "canonical_smiles"], 'smiles')[0].cid
                except:
                    pass
            pert_info.to_csv(pert_info_with_cid_path)

        # Load sig_info for both phases
        sig_info_1 = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]), sep="\t",
                               index_col="sig_id",
                               usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"])
        sig_info_2 = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[6]), sep="\t",
                                 index_col="sig_id",
                                 usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"])
        sig_info = pd.concat([sig_info_1, sig_info_2])

        # Convert time to float and add to sig_info
        sig_info['pert_itime_value'] = sig_info['pert_itime'].apply(self.get_time)
        # Convert concentrations to float and add to sig_info
        sig_info['pert_idose_value'] = sig_info['pert_idose'].apply(self.get_concentration)

        # Add cid to sig_info
        pert_to_cid_dict = pert_info['cid'].to_dict()
        sig_info['cid'] = sig_info['pert_id'].apply(
            lambda s: pert_to_cid_dict[s] if s in pert_to_cid_dict.keys() else -1)
        sig_info = sig_info[sig_info['cid'] != -1]

        return sig_info, landmark_gene_list

    def load_expr_data(self, phase):
        assert phase in ["phase1", "phase2"]
        if phase == "phase1":
            df_path = os.path.join(self.raw_dir, "dataframe_phase1.pkl")
            file_name = self.raw_file_names[5]
        else:
            df_path = os.path.join(self.raw_dir, "dataframe_phase2.pkl")
            file_name = self.raw_file_names[0]
        if os.path.isfile(df_path):
            pickle_in = open(df_path, "rb")
            expr_data = pickle.load(pickle_in)
        else:  # If the data has not been saved yet, parse the original file and save dataframe
            print("Parsing original data, only happens the first time...")
            expr_data = parse(os.path.join(self.raw_dir, file_name), rid=self.landmark_gene_list).data_df.T
            # Ensure that the order of columns corresponds to landmark_gene_list
            expr_data = expr_data[self.landmark_gene_list]
            # Remove rows that are not in sig_info
            expr_data = expr_data[expr_data.index.isin(self.sig_info.index)]

            # Save data
            pickle_out = open(df_path, "wb")
            pickle.dump(expr_data, pickle_out, protocol=2)
            pickle_out.close()
        return expr_data

    def get_concentration(self, s):
        if s.endswith('µM') or s.endswith('um'):
            return float(s[:-3])
        if s.endswith('nM'):
            return 0.001 * float(s[:-3])
        return -1

    def get_time(self, s):
        return float(s[:-2])

    def get_smiles(self, cids):
        unique_cid = np.unique(cids)
        unique_cid = list(unique_cid)

        dict_path = os.path.join(self.raw_dir, "cid_to_smiles_dict.npy")
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


if __name__ == '__main__':
    dataset = L1000()
    print("toto")

