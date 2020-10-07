from torch_geometric.data import Data
from recover.datasets.drugcomb_data import AbstractDrugComb
import numpy as np
import pandas as pd
import torch
import os


########################################################################################################################
# DrugComb dataset that uses the raw cell response
########################################################################################################################


class DrugComb(AbstractDrugComb):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super(DrugComb, self).__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)

    @property
    def processed_file_names(self):
        return ['drugcomb_data.pt']

    def process(self):
        # Load dataframes
        self._drugcomb_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), low_memory=False)
        self._drug_protein_link = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[2]), sep='\t')
        self._protein_protein_interactions = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[3]), sep=' ')

        # Get nodes
        self.cid_to_smiles_dict = self._get_smiles()
        drug_nodes, protein_nodes, cid_to_idx_dict, name_to_idx_dict, is_drug = self._get_nodes()

        # Build edges
        ppi_edge_idx, ppi_edge_attr = self._get_ppi_edges(name_to_idx_dict)  # PPI edges are both ways
        # DPI and DDI edges are one way only
        dpi_edge_idx, dpi_edge_attr = self._get_dpi_edges(cid_to_idx_dict, name_to_idx_dict)
        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, ddi_edge_response = self._get_ddi_edges(cid_to_idx_dict)

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
        data.ddi_edge_response = torch.tensor(ddi_edge_response, dtype=torch.float)

        # Add fingerprints attributes to data
        data.fp_radius = self.fp_radius
        data.fp_bits = self.fp_bits

        data.is_drug = torch.tensor(is_drug, dtype=torch.long)

        # Log transform concentrations
        data.ddi_edge_attr[:, 0] = torch.log(data.ddi_edge_attr[:, 0] + 1e-6)
        data.ddi_edge_attr[:, 1] = torch.log(data.ddi_edge_attr[:, 1] + 1e-6)

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_ddi_edges(self, cid_to_idx_dict):
        print('Processing drug drug interaction edges..')

        # Drop rows that do not have second drug
        self._drugcomb_data = self._drugcomb_data.dropna(axis=0, subset=['drug_col'])

        # Add categorical encoding of cell lines
        self._drugcomb_data['cell_line_cat'] = self._drugcomb_data['cell_line_name'].astype('category').cat.codes

        self._drugcomb_data['drug_row_idx'] = self._drugcomb_data['drug_row_cid'].\
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        self._drugcomb_data['drug_col_idx'] = self._drugcomb_data['drug_col_cid'].\
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        self._drugcomb_data = self._drugcomb_data[self._drugcomb_data['drug_row_idx'] != -1]
        self._drugcomb_data = self._drugcomb_data[self._drugcomb_data['drug_col_idx'] != -1]

        is_pairs_of_diff_drugs = self._drugcomb_data['drug_row_idx'] != self._drugcomb_data['drug_col_idx']
        self._drugcomb_data = self._drugcomb_data[is_pairs_of_diff_drugs]

        ddi_edge_idx = self._drugcomb_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
        ddi_edge_classes = self._drugcomb_data['cell_line_cat'].to_numpy()
        ddi_edge_attr = self._drugcomb_data[['conc_r', 'conc_c']].to_numpy()
        ddi_edge_response = self._drugcomb_data['inhibition'].to_numpy()

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, ddi_edge_response


########################################################################################################################
# DrugComb dataset that uses the raw cell response without PPI
########################################################################################################################


class DrugCombNoPPI(DrugComb):
    def __init__(self,  transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super().__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)

    def process(self):
        super().process()

    @property
    def processed_file_names(self):
        return ['drugcomb_data_no_ppi.pt']

    def _get_ppi_edges(self, name_to_idx_dict):
        return np.empty([2, 0]), np.empty([1, 0])


if __name__ == '__main__':
    dataset = DrugComb(fp_bits=1024, fp_radius=4)
    dataset_no_ppi = DrugCombNoPPI(fp_bits=1024, fp_radius=4)
