def specific_cell_line(cell_line_name):
    def _specific_cell_line(data):
        cell_line_idx = data.cell_line_name_to_idx[cell_line_name]
        matches = data.ddi_edge_classes == cell_line_idx

        data.ddi_edge_idx = data.ddi_edge_idx[:, matches]
        data.ddi_edge_attr = data.ddi_edge_attr[matches]
        data.ddi_edge_classes = data.ddi_edge_classes[matches]
        data.ddi_edge_css = data.ddi_edge_css[matches]
        data.ddi_edge_zip = data.ddi_edge_zip[matches]
        data.ddi_edge_bliss = data.ddi_edge_bliss[matches]
        data.ddi_edge_loewe = data.ddi_edge_loewe[matches]
        data.ddi_edge_hsa = data.ddi_edge_hsa[matches]

        return data

    return _specific_cell_line

