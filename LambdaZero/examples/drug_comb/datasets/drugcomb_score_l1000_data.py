from LambdaZero.examples.drug_comb.datasets.l1000_data import L1000
from LambdaZero.examples.drug_comb.datasets.drugcomb_data_score import DrugCombScore
import numpy as np
import pandas as pd


########################################################################################################################
# DrugComb dataset that uses the synergy scores and gene expression features
########################################################################################################################


class DrugCombScoreL1000(DrugCombScore):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super().__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)
        print("\t", self.data.ddi_edge_attr.shape[0], "pairs of expression profiles")
        print("\t", len(set(self.data.ddi_edge_classes.numpy())), "cell-lines")

    def process(self):
        l1000 = L1000()

        # Load metadata
        sig_info, landmark_gene_list = l1000.load_metadata()
        expr_data = pd.concat([l1000.load_expr_data("phase1"), l1000.load_expr_data("phase2")], sort=False)

        self.l1000_df = pd.concat((expr_data, sig_info), axis=1)  # Concat with metadata

        super().process()

    @property
    def processed_file_names(self):
        return ['drugcomb_score_l1000_data.pt']

    def _get_drug_nodes(self):
        drug_nodes = super()._get_drug_nodes()

        # Restrict ourselves to nodes which are in l1000
        common_cids = set(drug_nodes.index).intersection(self.l1000_df['cid'].unique())
        drug_nodes = drug_nodes.loc[common_cids]

        return drug_nodes

    def _get_ddi_edges(self, cid_to_idx_dict):
        print('Processing drug drug interaction edges')

        self._score_data['drug_row_idx'] = self._score_data['drug_row_cid']. \
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        self._score_data['drug_col_idx'] = self._score_data['drug_col_cid']. \
            apply(lambda cid: cid_to_idx_dict[cid] if cid in cid_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        self._score_data = self._score_data[self._score_data['drug_row_idx'] != -1]
        self._score_data = self._score_data[self._score_data['drug_col_idx'] != -1]

        is_pairs_of_diff_drugs = self._score_data['drug_row_idx'] != self._score_data['drug_col_idx']
        self._score_data = self._score_data[is_pairs_of_diff_drugs]

        # Add categorical encoding of cell lines
        self._score_data['cell_line_cat'] = self._score_data['cell_line_name'].astype('category').cat.codes

        # Get cell line dictionary
        cell_lines = self._score_data['cell_line_name'].astype('category').cat.categories
        cell_lines = dict((v, k) for k, v in enumerate(cell_lines))

        # Restrict ourselves to 24h incubation time
        self.l1000_df = self.l1000_df[self.l1000_df['pert_itime_value'] == 24]

        # Restrict ourselves to 10 uM concentrations
        self.l1000_df = self.l1000_df[self.l1000_df['pert_idose_value'] == 10.]

        self.l1000_df["cell_line_cat"] = self.l1000_df["cell_id"].apply(lambda s: cell_lines[s]
        if s in cell_lines.keys() else -1)

        # Drop examples whose cell line is not in DrugComb
        self.l1000_df = self.l1000_df[self.l1000_df["cell_line_cat"] != -1]
        self.l1000_df = self.l1000_df.dropna()

        gene_expr_dict = {(s[1]['cell_line_cat'], int(s[1]['cid'])):
                              s[1][list(self.l1000_df.columns[:978])]
                          for s in self.l1000_df.iterrows()}

        self._score_data['expr_row'] = \
            self._score_data.apply(lambda s: gene_expr_dict[(s['cell_line_cat'], s['drug_row_cid'])]
            if (s['cell_line_cat'], s['drug_row_cid']) in gene_expr_dict.keys() else -1, axis=1)

        self._score_data['expr_col'] = \
            self._score_data.apply(lambda s: gene_expr_dict[(s['cell_line_cat'], s['drug_col_cid'])]
            if (s['cell_line_cat'], s['drug_col_cid']) in gene_expr_dict.keys() else -1, axis=1)

        self._score_data = self._score_data[self._score_data['expr_row'].apply(lambda s: type(s) != int)]
        self._score_data = self._score_data[self._score_data['expr_col'].apply(lambda s: type(s) != int)]

        # Edit categorical encoding of remaining cell lines
        self._score_data['cell_line_cat'] = self._score_data['cell_line_name'].astype('category').cat.codes

        # Concatenate expression features with IC50s. IC50s are log transformed
        expr_row = np.array(self._score_data['expr_row'].apply(lambda s: s.to_numpy()).to_list())
        expr_row = np.concatenate((expr_row, np.log(self._score_data[['ic50_row']].to_numpy() + 1e-6)), axis=1)
        expr_col = np.array(self._score_data['expr_col'].apply(lambda s: s.to_numpy()).to_list())
        expr_col = np.concatenate((expr_col, np.log(self._score_data[['ic50_col']].to_numpy() + 1e-6)), axis=1)

        ddi_edge_idx = self._score_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
        ddi_edge_attr = np.concatenate((expr_row, expr_col), axis=1).astype(float)
        ddi_edge_classes = self._score_data['cell_line_cat'].to_numpy()
        ddi_edge_css = self._score_data['css'].to_numpy()
        ddi_edge_zip = self._score_data['synergy_zip'].to_numpy()
        ddi_edge_bliss = self._score_data['synergy_bliss'].to_numpy()
        ddi_edge_loewe = self._score_data['synergy_loewe'].to_numpy()
        ddi_edge_hsa = self._score_data['synergy_hsa'].to_numpy()

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, ddi_edge_css, \
               ddi_edge_zip, ddi_edge_bliss, ddi_edge_loewe, ddi_edge_hsa


########################################################################################################################
# DrugComb dataset that uses the synergy scores and gene expression features without PPI
########################################################################################################################


class DrugCombScoreL1000NoPPI(DrugCombScoreL1000):
    def __init__(self,  transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        super().__init__(transform, pre_transform, fp_bits, fp_radius, ppi_confidence_thres)

    def process(self):
        super().process()

    @property
    def processed_file_names(self):
        return ['drugcomb_score_l1000_data_no_ppi.pt']

    def _get_ppi_edges(self, name_to_idx_dict):
        return np.empty([2, 0]), np.empty([1, 0])


if __name__ == '__main__':
    dataset = DrugCombScoreL1000()
    dataset_no_ppi = DrugCombScoreL1000NoPPI()
