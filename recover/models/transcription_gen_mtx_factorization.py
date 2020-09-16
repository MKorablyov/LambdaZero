import torch
import torch.nn.functional as F
from torch.nn import init, Parameter, ModuleList
from recover.models.layers.relation_aware_linear import RelationAwareLinear

class TranscriptionGeneratorMatrixFactorizationModel(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()

        num_mtxs = int(config['num_mtxs'])
        mtx_inner_dim = int(config['mtx_inner_dim'])
        num_shared_mtxs = int(config['num_shared_mtxs']) if 'num_shared_mtxs' in config else 0
        num_cell_lines = torch.unique(data.cell_id_int).shape[0]
        num_drugs = torch.unique(data.cid).shape[0]

        self.input_vecs = Parameter(1/100 * torch.rand(num_cell_lines, mtx_inner_dim))

        modules = []
        cond = lambda x: not config['alternate_mtx_types'] or x % 2 == 0
        for i in range(num_mtxs):
            out_dim = mtx_inner_dim if i != num_mtxs - 1 else data.gene_expr.shape[1]
            num_relations = num_drugs if cond(i) else num_cell_lines

            module = None
            if i < num_mtxs - num_shared_mtxs:
                module = RelationAwareLinear(num_relations, mtx_inner_dim, out_dim)
            else:
                module = torch.nn.Linear(mtx_inner_dim, out_dim)

            module.is_drug_lyr = cond(i)

            modules.append(module)

        self.mtx_modules = ModuleList(modules)
        self.dropout = torch.nn.Dropout(config['dropout_prob'])

        self.film = None
        if config['use_film']:
            film_vecs_sub = 1 if num_mtxs == 2 else 2
            num_film_vecs = max(0, num_mtxs - 1)

            self.film = Parameter(1/100 * torch.rand(num_film_vecs, num_cell_lines))

    def forward(self, cell_line_idx, drug_idx):
        out = self.input_vecs[cell_line_idx]

        for i in range(len(self.mtx_modules)):
            rel_idx = drug_idx if self.mtx_modules[i].is_drug_lyr else cell_line_idx
            args = [out]
            if isinstance(self.mtx_modules[i], RelationAwareLinear):
                args.append(rel_idx)

            out = self.mtx_modules[i](*args)

            if i != len(self.mtx_modules) - 1:
                if self.film is not None:
                    out = out * self.film[i, cell_line_idx].unsqueeze(dim=1)

                out = self.dropout(out)
                out = F.relu(out)

        return out

