from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch
import os
from LambdaZero.examples.drug_comb.datasets.drugcomb_data import AbstractDrugComb


########################################################################################################################
# DrugComb dataset that uses the synergy scores
########################################################################################################################


class DrugCombScore(AbstractDrugComb):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super(DrugCombScore, self).__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)

    @property
    def processed_file_names(self):
        return ['drugcomb_score_data.pt']

    def process(self):

        # Load dataframes
        self._drugcomb_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), low_memory=False)
        self._score_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]), low_memory=False)
        self._drug_protein_link = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[2]), sep='\t')
        self._protein_protein_interactions = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[3]), sep=' ')

        # Add cid info to the score dataframe
        self.add_cid_to_score_df()

        # Get nodes
        self.cid_to_smiles_dict = self._get_smiles()
        drug_nodes, protein_nodes, cid_to_idx_dict, name_to_idx_dict, is_drug = self._get_nodes()

        # Build edges
        ppi_edge_idx, ppi_edge_attr = self._get_ppi_edges(name_to_idx_dict)  # PPI edges are both ways
        # DPI and DDI edges are one way only
        dpi_edge_idx, dpi_edge_attr = self._get_dpi_edges(cid_to_idx_dict, name_to_idx_dict)
        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, ddi_edge_css, ddi_edge_zip, ddi_edge_bliss, \
        ddi_edge_loewe, ddi_edge_hsa = self._get_ddi_edges(cid_to_idx_dict)

        data = Data(x_drugs=torch.tensor(drug_nodes.to_numpy(), dtype=torch.float),
                    x_prots=torch.tensor(protein_nodes.to_numpy(), dtype=torch.float))

        # Add ppi attributes to data
        data.ppi_edge_idx = torch.tensor(ppi_edge_idx, dtype=torch.long)
        data.ppi_edge_attr = torch.tensor(ppi_edge_attr, dtype=torch.float)
        # Add dpi attributes to data
        data.dpi_edge_idx = torch.tensor(dpi_edge_idx, dtype=torch.long)
        data.dpi_edge_attr = torch.tensor(dpi_edge_attr, dtype=torch.float)
        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.long)
        data.ddi_edge_css = torch.tensor(ddi_edge_css, dtype=torch.float)
        data.ddi_edge_zip = torch.tensor(ddi_edge_zip, dtype=torch.float)
        data.ddi_edge_bliss = torch.tensor(ddi_edge_bliss, dtype=torch.float)
        data.ddi_edge_loewe = torch.tensor(ddi_edge_loewe, dtype=torch.float)
        data.ddi_edge_hsa = torch.tensor(ddi_edge_hsa, dtype=torch.float)

        # Add fingerprints attributes to data
        data.fp_radius = self.fp_radius
        data.fp_bits = self.fp_bits

        data.is_drug = torch.tensor(is_drug, dtype=torch.long)
        data.cid_to_idx_dict = cid_to_idx_dict

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def add_cid_to_score_df(self):

        drug_cid_row = self._drugcomb_data[['drug_row', 'drug_row_cid']]
        drug_cid_col = self._drugcomb_data[['drug_col', 'drug_col_cid']]
        drug_cid_row.columns = ['drug', 'cid']
        drug_cid_col.columns = ['drug', 'cid']
        drug_cid = pd.concat((drug_cid_row, drug_cid_col)).drop_duplicates().dropna()
        drug_cid = drug_cid.set_index('drug', drop=True).to_dict()['cid']

        self._score_data['drug_row_cid'] = self._score_data['drug_row'].apply(lambda s: drug_cid[s])
        self._score_data['drug_col_cid'] = self._score_data['drug_col'].apply(lambda s: drug_cid[s]
        if s in drug_cid.keys() else -1)

        self._score_data = self._score_data[self._score_data['drug_col_cid'] != -1]

    def _get_ddi_edges(self, cid_to_idx_dict):
        print('Processing drug drug interaction edges..')

        # Add categorical encoding of cell lines
        self._score_data['cell_line_cat'] = self._score_data['cell_line_name'].astype('category').cat.codes

        self._score_data['drug_row_idx'] = self._score_data['drug_row_cid']. \
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        self._score_data['drug_col_idx'] = self._score_data['drug_col_cid']. \
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        self._score_data = self._score_data[self._score_data['drug_row_idx'] != -1]
        self._score_data = self._score_data[self._score_data['drug_col_idx'] != -1]

        is_pairs_of_diff_drugs = self._score_data['drug_row_idx'] != self._score_data['drug_col_idx']
        self._score_data = self._score_data[is_pairs_of_diff_drugs]

        ddi_edge_idx = self._score_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
        ddi_edge_classes = self._score_data['cell_line_cat'].to_numpy()
        ddi_edge_css = self._score_data['css'].to_numpy()
        ddi_edge_zip = self._score_data['synergy_zip'].to_numpy()
        ddi_edge_bliss = self._score_data['synergy_bliss'].to_numpy()
        ddi_edge_loewe = self._score_data['synergy_loewe'].to_numpy()
        ddi_edge_hsa = self._score_data['synergy_hsa'].to_numpy()
        ddi_edge_attr = self._score_data[['ic50_row', 'ic50_col']].to_numpy()

        # Log transform ic50s
        ddi_edge_attr = np.log(ddi_edge_attr + 1e-6)

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, ddi_edge_css, \
               ddi_edge_zip, ddi_edge_bliss, ddi_edge_loewe, ddi_edge_hsa


class DrugCombScoreNoPPI(DrugCombScore):
    def __init__(self,  transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super().__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)

    def process(self):
        super().process()

    @property
    def processed_file_names(self):
        return ['drugcomb_score_data_no_ppi.pt']

    def _get_ppi_edges(self, name_to_idx_dict):
        return np.empty([2, 0]), np.empty([1, 0])


if __name__ == '__main__':
    dataset = DrugCombScore(fp_bits=1024, fp_radius=4)
    dataset_no_ppi = DrugCombScoreNoPPI(fp_bits=1024, fp_radius=4)
